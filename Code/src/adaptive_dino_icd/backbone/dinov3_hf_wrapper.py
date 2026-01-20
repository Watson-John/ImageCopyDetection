"""
DINOv3 HuggingFace Wrapper.

Loads DINOv3 from HuggingFace with proper error handling.
NO automatic fallback to other models - fails with clear instructions if unavailable.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the DINOv3 model cannot be loaded."""

    pass


class Dinov3HFWrapper(nn.Module):
    """
    Wrapper for DINOv3 backbone from HuggingFace.

    Load strategy:
    1. If offline_stub=True: Use timm ViT (for testing only)
    2. Otherwise: Try to load from HuggingFace transformers
    3. If unavailable: Raise ModelNotFoundError with clear instructions

    NO automatic fallback to DINOv2 or other models.

    Args:
        model_name: HuggingFace model name (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
        offline_stub: If True, use timm ViT stub for testing
        pretrained: Whether to load pretrained weights
        embed_dim: Expected embedding dimension (for validation)

    Example:
        # Production use (requires HF access):
        wrapper = Dinov3HFWrapper(offline_stub=False)

        # Testing (no network required):
        wrapper = Dinov3HFWrapper(offline_stub=True)

        # Forward pass
        outputs = wrapper(images)
        global_feats = outputs["global_feats"]  # (B, D)
        local_feats = outputs["local_feats"]    # (B, N, D)
    """

    DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def __init__(
        self,
        model_name: str = DINOV3_MODEL_NAME,
        offline_stub: bool = False,
        pretrained: bool = True,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.model_name = model_name
        self.offline_stub = offline_stub
        self.pretrained = pretrained
        self.embed_dim = embed_dim

        if offline_stub:
            self._load_stub_model()
        else:
            self._load_hf_model()

    def _load_stub_model(self) -> None:
        """Load timm ViT as offline stub for testing."""
        logger.warning(
            "Loading OFFLINE STUB model (timm ViT). "
            "This is for testing only - not suitable for production!"
        )

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for offline stub mode. "
                "Install with: pip install timm"
            )

        # Use ViT-Base/16 from timm
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=self.pretrained,
            num_classes=0,  # Remove classification head
        )

        self.is_stub = True
        self._model_type = "timm"

        # Verify embed_dim
        if hasattr(self.model, "embed_dim"):
            actual_dim = self.model.embed_dim
            if actual_dim != self.embed_dim:
                logger.warning(
                    f"Stub model embed_dim ({actual_dim}) differs from expected ({self.embed_dim})"
                )
                self.embed_dim = actual_dim

    def _load_hf_model(self) -> None:
        """Load DINOv3 from HuggingFace."""
        logger.info(f"Loading DINOv3 from HuggingFace: {self.model_name}")

        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace model loading. "
                "Install with: pip install transformers"
            )

        try:
            # First try to load config to check if model exists
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load the model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                config=config,
                trust_remote_code=True,
            )

            self.is_stub = False
            self._model_type = "transformers"

            # Get embed_dim from config
            if hasattr(config, "hidden_size"):
                self.embed_dim = config.hidden_size
            elif hasattr(config, "embed_dim"):
                self.embed_dim = config.embed_dim

            logger.info(f"Successfully loaded {self.model_name} (embed_dim={self.embed_dim})")

        except Exception as e:
            error_msg = self._format_model_not_found_error(e)
            raise ModelNotFoundError(error_msg)

    def _format_model_not_found_error(self, original_error: Exception) -> str:
        """Format a helpful error message when model cannot be loaded."""
        return f"""
================================================================================
DINOv3 MODEL NOT FOUND
================================================================================

Could not load model: {self.model_name}

Original error: {original_error}

Please ensure you have:

1. ACCESS TO THE MODEL ON HUGGINGFACE
   - The model may require authentication or approval
   - Visit: https://huggingface.co/{self.model_name}

2. SET HF_TOKEN ENVIRONMENT VARIABLE (if required)
   - Get your token from: https://huggingface.co/settings/tokens
   - Set it: export HF_TOKEN=your_token_here
   - Or in Python: huggingface_hub.login(token="your_token_here")

3. ALTERNATIVELY, USE OFFLINE STUB FOR TESTING
   - Run with: --config configs/phase1_offline_stub.yaml
   - Or set: backbone.offline_stub=true in your config

NOTE: This system does NOT automatically fall back to other models.
      You must explicitly configure the model you want to use.

================================================================================
"""

    def forward(
        self,
        images: torch.Tensor,
        return_all_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DINOv3 backbone.

        Args:
            images: Input images (B, C, H, W), normalized
            return_all_tokens: If True, return all patch tokens; if False, only CLS

        Returns:
            Dictionary with:
                - global_feats: (B, D) CLS token or mean-pooled features
                - local_feats: (B, N, D) patch tokens (if return_all_tokens=True)
        """
        if self._model_type == "timm":
            return self._forward_timm(images, return_all_tokens)
        else:
            return self._forward_transformers(images, return_all_tokens)

    def _forward_timm(
        self,
        images: torch.Tensor,
        return_all_tokens: bool,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for timm model."""
        # timm ViT forward_features returns (B, N+1, D) with CLS token
        features = self.model.forward_features(images)  # (B, N+1, D)

        # Split CLS and patch tokens
        cls_token = features[:, 0]  # (B, D)
        patch_tokens = features[:, 1:]  # (B, N, D)

        result = {"global_feats": cls_token}

        if return_all_tokens:
            result["local_feats"] = patch_tokens

        return result

    def _forward_transformers(
        self,
        images: torch.Tensor,
        return_all_tokens: bool,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for transformers model."""
        # Use output_hidden_states to get all features
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )

        # Try different attribute names for different model architectures
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state  # (B, N+1, D)
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]  # Last layer
        else:
            raise RuntimeError(
                f"Cannot extract features from model output. "
                f"Available attributes: {dir(outputs)}"
            )

        # Check if there's a CLS token (position 0)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            global_feats = outputs.pooler_output  # (B, D)
            patch_tokens = hidden_states[:, 1:]  # (B, N, D)
        else:
            # Assume CLS token at position 0
            global_feats = hidden_states[:, 0]  # (B, D)
            patch_tokens = hidden_states[:, 1:]  # (B, N, D)

        result = {"global_feats": global_feats}

        if return_all_tokens:
            result["local_feats"] = patch_tokens

        return result

    def get_intermediate_features(
        self,
        images: torch.Tensor,
        layer_indices: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate layer features for multi-scale analysis.

        Args:
            images: Input images (B, C, H, W)
            layer_indices: Which layers to return (default: last 4 layers)

        Returns:
            Dictionary with layer index -> features mapping
        """
        if self._model_type == "timm":
            return self._get_intermediate_timm(images, layer_indices)
        else:
            return self._get_intermediate_transformers(images, layer_indices)

    def _get_intermediate_timm(
        self,
        images: torch.Tensor,
        layer_indices: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get intermediate features from timm model."""
        # timm models have different ways to get intermediate features
        # This is a simplified version - may need adjustment for specific models
        features = {}

        # Get features from forward_features
        x = self.model.patch_embed(images)
        x = self.model._pos_embed(x)

        if layer_indices is None:
            # Default: last 4 layers
            n_blocks = len(self.model.blocks)
            layer_indices = tuple(range(max(0, n_blocks - 4), n_blocks))

        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in layer_indices:
                features[f"layer_{i}"] = x.clone()

        return features

    def _get_intermediate_transformers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get intermediate features from transformers model."""
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # Tuple of (B, N+1, D)

        if layer_indices is None:
            # Default: last 4 layers
            layer_indices = tuple(range(max(0, len(hidden_states) - 4), len(hidden_states)))

        features = {}
        for i in layer_indices:
            if i < len(hidden_states):
                features[f"layer_{i}"] = hidden_states[i]

        return features

    @property
    def num_features(self) -> int:
        """Return embedding dimension."""
        return self.embed_dim

    @property
    def patch_size(self) -> int:
        """Return patch size used by the model."""
        return 16  # Standard for ViT-B/16
