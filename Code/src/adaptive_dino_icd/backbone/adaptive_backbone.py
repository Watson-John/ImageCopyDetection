"""
Adaptive Backbone combining DINOv3 with APT tokenization.

Orchestrates the full pipeline:
EntropyScorer → PatchSelector → PatchAggregator → DINOv3 Wrapper
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_dino_icd.apt import EntropyScorer, PatchSelector, PatchAggregator, SequencePacker
from adaptive_dino_icd.backbone.dinov3_hf_wrapper import Dinov3HFWrapper
from adaptive_dino_icd.backbone.rope_adapter import RoPEAdapter
from adaptive_dino_icd.utils.config import BackboneConfig, APTConfig

logger = logging.getLogger(__name__)


class AdaptiveBackbone(nn.Module):
    """
    Adaptive backbone combining APT tokenization with DINOv3.

    Pipeline:
    1. EntropyScorer: Compute multi-scale entropy maps
    2. PatchSelector: Select adaptive patches based on entropy
    3. PatchAggregator: Embed variable-sized patches into tokens
    4. RoPEAdapter: Compute position encodings (Phase 2: inject into DINOv3)
    5. Dinov3HFWrapper: Process tokens through transformer

    Args:
        backbone_config: Configuration for DINOv3 wrapper
        apt_config: Configuration for APT components
        use_adaptive_patches: If False, use standard uniform patches

    Example:
        config = Config.from_dict(yaml.load("config.yaml"))
        model = AdaptiveBackbone(config.backbone, config.apt)

        outputs = model(images)
        global_feats = outputs["global_feats"]  # (B, D)
        local_feats = outputs["local_feats"]    # (B, N_max, D)
        local_coords = outputs["local_coords"]  # (B, N_max, 2)
    """

    def __init__(
        self,
        backbone_config: Optional[BackboneConfig] = None,
        apt_config: Optional[APTConfig] = None,
        use_adaptive_patches: bool = True,
    ):
        super().__init__()

        # Use defaults if configs not provided
        self.backbone_config = backbone_config or BackboneConfig()
        self.apt_config = apt_config or APTConfig()
        self.use_adaptive_patches = use_adaptive_patches

        # Initialize components
        self._build_components()

        logger.info(
            f"AdaptiveBackbone initialized: "
            f"adaptive_patches={use_adaptive_patches}, "
            f"embed_dim={self.embed_dim}, "
            f"offline_stub={self.backbone_config.offline_stub}"
        )

    def _build_components(self) -> None:
        """Build all sub-components."""
        # DINOv3 backbone
        self.backbone = Dinov3HFWrapper(
            model_name=self.backbone_config.model_name,
            offline_stub=self.backbone_config.offline_stub,
            pretrained=self.backbone_config.pretrained,
            embed_dim=self.backbone_config.embed_dim,
        )

        self.embed_dim = self.backbone.embed_dim

        # APT components (only if using adaptive patches)
        if self.use_adaptive_patches:
            self.entropy_scorer = EntropyScorer(
                scales=self.apt_config.entropy_scales,
                normalize=self.apt_config.normalize_entropy,
            )

            self.patch_selector = PatchSelector(
                min_patch_size=self.apt_config.min_patch_size,
                max_patch_size=self.apt_config.max_patch_size,
                entropy_thresholds=self.apt_config.entropy_thresholds,
            )

            self.patch_aggregator = PatchAggregator(
                embed_dim=self.embed_dim,
                smallest_patch_size=self.apt_config.smallest_patch_size,
            )

            self.sequence_packer = SequencePacker()
        else:
            self.entropy_scorer = None
            self.patch_selector = None
            self.patch_aggregator = None
            self.sequence_packer = None

        # RoPE adapter (computes coords even if not injecting)
        self.rope_adapter = RoPEAdapter(
            embed_dim=self.embed_dim,
        )

        # Optional: freeze backbone
        if self.backbone_config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")

    def forward(
        self,
        images: torch.Tensor,
        return_entropy: bool = False,
        return_patches: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through adaptive backbone.

        Args:
            images: Input images (B, C, H, W), expected normalized to [0, 1] or [-1, 1]
            return_entropy: If True, include entropy maps in output
            return_patches: If True, include patch metadata in output

        Returns:
            Dictionary with:
                - global_feats: (B, D) global feature vector
                - local_feats: (B, N_max, D) padded local features
                - local_coords: (B, N_max, 2) padded coordinates in [-1, 1]
                - token_counts: (B,) number of tokens per image
                - attention_mask: (B, N_max) boolean mask for valid tokens
                - entropy_maps: (optional) Dict of entropy maps
                - patches: (optional) List of PatchRegion lists
        """
        B, C, H, W = images.shape

        if self.use_adaptive_patches:
            return self._forward_adaptive(
                images, return_entropy=return_entropy, return_patches=return_patches
            )
        else:
            return self._forward_standard(images)

    def _forward_adaptive(
        self,
        images: torch.Tensor,
        return_entropy: bool = False,
        return_patches: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive patch selection."""
        B, C, H, W = images.shape
        device = images.device

        # Step 1: Compute entropy maps
        entropy_maps = self.entropy_scorer(images)

        # Step 2: Select patches based on entropy
        patches = self.patch_selector(entropy_maps, image_size=(H, W))

        # Step 3: Embed patches into tokens
        aggregated = self.patch_aggregator.forward_padded(images, patches)

        tokens = aggregated["tokens"]  # (B, N_max, D)
        coords = aggregated["coords"]  # (B, N_max, 2)
        attention_mask = aggregated["attention_mask"]  # (B, N_max)
        token_counts = aggregated["token_counts"]  # (B,)

        # Step 4: Process RoPE (Phase 1: compute but don't inject)
        rope_output = self.rope_adapter(coords)

        # Step 5: Get features from backbone
        # In Phase 1, we use standard backbone forward and combine with APT tokens
        backbone_output = self.backbone(images)

        # Use backbone's global features as primary global representation
        global_feats = backbone_output["global_feats"]  # (B, D)

        # For local features, we use the APT-embedded tokens
        # In Phase 2, these would go through the transformer with RoPE
        local_feats = tokens  # (B, N_max, D)

        # Build result
        result = {
            "global_feats": global_feats,
            "local_feats": local_feats,
            "local_coords": coords,
            "token_counts": token_counts,
            "attention_mask": attention_mask,
        }

        if return_entropy:
            result["entropy_maps"] = entropy_maps

        if return_patches:
            result["patches"] = patches

        return result

    def _forward_standard(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with standard uniform patches (no APT)."""
        B, C, H, W = images.shape
        device = images.device

        # Get features from backbone
        backbone_output = self.backbone(images)

        global_feats = backbone_output["global_feats"]  # (B, D)
        local_feats = backbone_output.get("local_feats")  # (B, N, D)

        if local_feats is None:
            # Create dummy local features if backbone doesn't return them
            N = (H // 16) * (W // 16)  # Standard patch count
            local_feats = torch.zeros(B, N, self.embed_dim, device=device)

        N = local_feats.shape[1]

        # Compute standard grid coordinates
        coords = self.rope_adapter.compute_coords_from_patches(
            patches_per_image=N,
            image_size=(H, W),
            patch_size=16,
        ).to(device)
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # All tokens valid for standard patches
        attention_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        token_counts = torch.full((B,), N, device=device)

        return {
            "global_feats": global_feats,
            "local_feats": local_feats,
            "local_coords": coords,
            "token_counts": token_counts,
            "attention_mask": attention_mask,
        }

    def get_global_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get only global features (more efficient if local not needed).

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Global features (B, D)
        """
        backbone_output = self.backbone(images, return_all_tokens=False)
        return backbone_output["global_feats"]

    def extract_features(
        self,
        images: torch.Tensor,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features suitable for similarity computation.

        Args:
            images: Input images (B, C, H, W)
            normalize: Whether to L2-normalize features

        Returns:
            Dictionary with normalized global and local features
        """
        outputs = self.forward(images)

        global_feats = outputs["global_feats"]
        local_feats = outputs["local_feats"]

        if normalize:
            global_feats = F.normalize(global_feats, p=2, dim=-1)
            # Normalize each local token
            local_feats = F.normalize(local_feats, p=2, dim=-1)

        return {
            "global_feats": global_feats,
            "local_feats": local_feats,
            "local_coords": outputs["local_coords"],
            "attention_mask": outputs["attention_mask"],
        }

    @property
    def num_features(self) -> int:
        """Return embedding dimension."""
        return self.embed_dim

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
