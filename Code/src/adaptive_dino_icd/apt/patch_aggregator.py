"""
Patch Aggregator for Adaptive Patch Tokenization.

Embeds patches of various sizes into fixed-dimensional tokens.
Small patches use direct linear projection; larger patches use
hierarchical embedding with ZeroMLP fusion.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_dino_icd.apt.patch_selector import PatchRegion


class ZeroMLP(nn.Module):
    """
    Zero-initialized MLP for patch fusion.

    All weights are initialized to zero, allowing the network to start
    as an identity-like operation and gradually learn the fusion function.
    This prevents disrupting pretrained representations early in training.

    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension
        num_layers: Number of linear layers (default: 2)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        current_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))

        self.layers = nn.Sequential(*layers)

        # Zero-initialize all weights
        self._zero_init()

    def _zero_init(self):
        """Initialize all linear layer weights to zero."""
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through zero-initialized MLP.

        Args:
            x: Input tensor (*, in_dim)

        Returns:
            Output tensor (*, out_dim)
        """
        return self.layers(x)


class PatchAggregator(nn.Module):
    """
    Aggregate variable-sized patches into fixed-dimensional token embeddings.

    Strategy:
    - Small patches (== smallest_patch_size): Direct linear projection
    - Large patches: Resize to standard size, extract sub-patches, embed each,
                    apply conv downsample + ZeroMLP fusion

    Args:
        embed_dim: Output embedding dimension
        smallest_patch_size: Base patch size for embedding (typically 16 for ViT)
        hidden_dim: Hidden dimension for ZeroMLP (default: 4 * embed_dim)
        image_size: Expected input image size for positional calculations

    Example:
        aggregator = PatchAggregator(embed_dim=768, smallest_patch_size=16)
        tokens, coords = aggregator(images, patches)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        smallest_patch_size: int = 16,
        hidden_dim: Optional[int] = None,
        image_size: int = 224,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.smallest_patch_size = smallest_patch_size
        self.hidden_dim = hidden_dim or 4 * embed_dim
        self.image_size = image_size

        # Patch embedding for smallest patches (standard ViT-style)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=smallest_patch_size,
            stride=smallest_patch_size,
        )

        # Resize embedding: embed patches that have been resized to standard size
        self.resize_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=smallest_patch_size,
            stride=smallest_patch_size,
        )

        # ZeroMLP for fusing sub-patch embeddings
        self.fusion_mlp = ZeroMLP(
            in_dim=embed_dim,
            hidden_dim=self.hidden_dim,
            out_dim=embed_dim,
        )

        # Conv for downsampling sub-patch grid to single token
        # Applied before ZeroMLP when we have multiple sub-patches
        self.downsample_conv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
        )

        # Layer norm for token normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        patches: List[List[PatchRegion]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Embed patches into tokens with normalized coordinates.

        Args:
            images: Input images (B, C, H, W)
            patches: List of patch lists from PatchSelector, one per image

        Returns:
            tokens: List of token tensors, one per image (each: N_i, D)
            coords: List of coordinate tensors, one per image (each: N_i, 2)
                   Coordinates are in [-1, 1] normalized space
        """
        B, C, H, W = images.shape
        device = images.device

        all_tokens = []
        all_coords = []

        for b in range(B):
            img = images[b]  # (C, H, W)
            img_patches = patches[b]

            if len(img_patches) == 0:
                # No patches - return empty tensors
                all_tokens.append(torch.zeros(0, self.embed_dim, device=device))
                all_coords.append(torch.zeros(0, 2, device=device))
                continue

            tokens = []
            coords = []

            for patch in img_patches:
                # Extract patch region
                patch_img = img[
                    :,
                    patch.y : patch.y + patch.h,
                    patch.x : patch.x + patch.w,
                ]  # (C, h, w)

                # Embed based on patch size
                token = self._embed_patch(patch_img, patch)
                tokens.append(token)

                # Compute normalized coordinates
                x_norm, y_norm = patch.to_normalized_coords(H, W)
                coords.append(torch.tensor([x_norm, y_norm], device=device))

            # Stack tokens and coords
            tokens = torch.stack(tokens, dim=0)  # (N, D)
            tokens = self.norm(tokens)
            coords = torch.stack(coords, dim=0)  # (N, 2)

            all_tokens.append(tokens)
            all_coords.append(coords)

        return all_tokens, all_coords

    def _embed_patch(
        self,
        patch_img: torch.Tensor,
        patch: PatchRegion,
    ) -> torch.Tensor:
        """
        Embed a single patch into a token.

        Args:
            patch_img: Patch image (C, h, w)
            patch: PatchRegion with metadata

        Returns:
            Token embedding (D,)
        """
        C, h, w = patch_img.shape

        if h == self.smallest_patch_size and w == self.smallest_patch_size:
            # Direct embedding for smallest patches
            patch_img = patch_img.unsqueeze(0)  # (1, C, h, w)
            token = self.patch_embed(patch_img)  # (1, D, 1, 1)
            token = token.flatten()  # (D,)

        elif h < self.smallest_patch_size or w < self.smallest_patch_size:
            # Patch smaller than minimum - resize up and embed
            patch_img = F.interpolate(
                patch_img.unsqueeze(0),
                size=(self.smallest_patch_size, self.smallest_patch_size),
                mode="bilinear",
                align_corners=False,
            )
            token = self.resize_embed(patch_img)  # (1, D, 1, 1)
            token = token.flatten()  # (D,)

        else:
            # Large patch - hierarchical embedding
            token = self._embed_large_patch(patch_img)

        return token

    def _embed_large_patch(self, patch_img: torch.Tensor) -> torch.Tensor:
        """
        Embed a large patch using hierarchical sub-patch embedding.

        Strategy:
        1. Resize to standardized size (multiple of smallest_patch_size)
        2. Extract sub-patches at smallest_patch_size
        3. Embed each sub-patch
        4. Pool/aggregate sub-patch embeddings
        5. Apply ZeroMLP fusion

        Args:
            patch_img: Large patch image (C, h, w)

        Returns:
            Token embedding (D,)
        """
        C, h, w = patch_img.shape

        # Resize to standard grid size
        target_h = max(self.smallest_patch_size, (h // self.smallest_patch_size) * self.smallest_patch_size)
        target_w = max(self.smallest_patch_size, (w // self.smallest_patch_size) * self.smallest_patch_size)

        # Ensure at least 2x2 sub-patches for large patches
        target_h = max(target_h, 2 * self.smallest_patch_size)
        target_w = max(target_w, 2 * self.smallest_patch_size)

        resized = F.interpolate(
            patch_img.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )  # (1, C, target_h, target_w)

        # Embed all sub-patches at once using conv
        sub_embeddings = self.resize_embed(resized)  # (1, D, n_h, n_w)

        # Global average pool over spatial dimensions
        pooled = F.adaptive_avg_pool2d(sub_embeddings, (1, 1))  # (1, D, 1, 1)
        pooled = pooled.flatten()  # (D,)

        # Apply ZeroMLP fusion
        token = pooled + self.fusion_mlp(pooled)

        return token

    def forward_padded(
        self,
        images: torch.Tensor,
        patches: List[List[PatchRegion]],
    ) -> Dict[str, torch.Tensor]:
        """
        Embed patches and return padded tensors with attention mask.

        This is more efficient for batched processing with transformers.

        Args:
            images: Input images (B, C, H, W)
            patches: List of patch lists from PatchSelector

        Returns:
            Dictionary with:
                - tokens: Padded token tensor (B, N_max, D)
                - coords: Padded coordinate tensor (B, N_max, 2)
                - attention_mask: Boolean mask (B, N_max), True for valid tokens
                - token_counts: Number of tokens per image (B,)
        """
        tokens_list, coords_list = self.forward(images, patches)

        B = len(tokens_list)
        device = images.device

        # Find max token count
        token_counts = [t.shape[0] for t in tokens_list]
        max_tokens = max(token_counts) if token_counts else 1

        # Pad tokens and coords
        padded_tokens = torch.zeros(B, max_tokens, self.embed_dim, device=device)
        padded_coords = torch.zeros(B, max_tokens, 2, device=device)
        attention_mask = torch.zeros(B, max_tokens, dtype=torch.bool, device=device)

        for b, (tokens, coords, count) in enumerate(zip(tokens_list, coords_list, token_counts)):
            if count > 0:
                padded_tokens[b, :count] = tokens
                padded_coords[b, :count] = coords
                attention_mask[b, :count] = True

        return {
            "tokens": padded_tokens,
            "coords": padded_coords,
            "attention_mask": attention_mask,
            "token_counts": torch.tensor(token_counts, device=device),
        }
