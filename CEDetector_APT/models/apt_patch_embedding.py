"""
APT Patch Embedding Module
Handles embedding of patches of different sizes into uniform token space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ZeroMLP(nn.Module):
    """
    Zero-initialized MLP for gradual integration of high-resolution details.
    Based on ControlNet's zero-convolution idea.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Initialize weights to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class APTPatchEmbedding(nn.Module):
    """
    Patch embedding module that handles variable patch sizes.
    Combines resizing with sub-patch aggregation for better performance.
    """

    def __init__(
        self,
        base_patch_size: int = 16,
        num_scales: int = 3,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            base_patch_size: Smallest patch size
            num_scales: Number of patch scales
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.base_patch_size = base_patch_size
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Base patch embedding layer (for smallest patches)
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=base_patch_size,
            stride=base_patch_size
        )

        # Convolutional aggregation layers for each scale
        self.conv_aggregators = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            for _ in range(num_scales - 1)
        ])

        # Zero-initialized MLPs for combining resized and aggregated embeddings
        self.zero_mlps = nn.ModuleList([
            ZeroMLP(embed_dim)
            for _ in range(num_scales - 1)
        ])

    def embed_base_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Embed a patch of base size using the patch embedding layer.

        Args:
            patch: Patch of shape (B, C, base_patch_size, base_patch_size)

        Returns:
            Embedding of shape (B, embed_dim)
        """
        # Add batch dimension if needed
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)

        embedding = self.patch_embed(patch)  # (B, embed_dim, 1, 1)
        embedding = embedding.flatten(2).squeeze(-1)  # (B, embed_dim)

        return embedding

    def embed_multi_scale_patch(
        self,
        patch: torch.Tensor,
        scale_idx: int
    ) -> torch.Tensor:
        """
        Embed a patch of arbitrary scale.
        For larger patches, both resize and aggregate sub-patches.

        Args:
            patch: Patch of shape (B, C, H, W) or (C, H, W)
            scale_idx: Scale index (0 = base size, 1 = 2x base, etc.)

        Returns:
            Embedding of shape (B, embed_dim)
        """
        # Add batch dimension if needed
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)

        if scale_idx == 0:
            # Base scale - direct embedding
            return self.embed_base_patch(patch)

        # For larger patches, combine resizing and sub-patch aggregation
        B, C, H, W = patch.shape

        # 1. Resize to base size and embed
        resized_patch = F.interpolate(
            patch,
            size=(self.base_patch_size, self.base_patch_size),
            mode='bilinear',
            align_corners=False
        )
        resized_embedding = self.embed_base_patch(resized_patch)  # (B, embed_dim)

        # 2. Split into sub-patches and aggregate
        # Number of sub-patches = 2^scale_idx in each dimension
        num_subpatches = 2 ** scale_idx
        sub_patch_size = self.base_patch_size

        # Unfold to get sub-patches
        sub_patches = F.unfold(
            patch,
            kernel_size=sub_patch_size,
            stride=sub_patch_size
        )  # (B, C*sub_patch_size^2, num_subpatches^2)

        # Reshape to (B, C, sub_patch_size, sub_patch_size, num_subpatches, num_subpatches)
        sub_patches = sub_patches.view(
            B, C, sub_patch_size, sub_patch_size, num_subpatches, num_subpatches
        )

        # Embed each sub-patch
        sub_embeddings = []
        for i in range(num_subpatches):
            for j in range(num_subpatches):
                sub_patch = sub_patches[:, :, :, :, i, j]  # (B, C, sub_patch_size, sub_patch_size)
                sub_emb = self.embed_base_patch(sub_patch)  # (B, embed_dim)
                sub_embeddings.append(sub_emb)

        # Stack sub-embeddings: (B, num_subpatches^2, embed_dim)
        sub_embeddings = torch.stack(sub_embeddings, dim=1)

        # Reshape to spatial grid: (B, embed_dim, num_subpatches, num_subpatches)
        sub_embeddings = sub_embeddings.permute(0, 2, 1).view(
            B, self.embed_dim, num_subpatches, num_subpatches
        )

        # Apply convolutional aggregation scale_idx times
        aggregated = sub_embeddings
        for i in range(scale_idx):
            aggregated = self.conv_aggregators[i](aggregated)

        # Flatten: (B, embed_dim, 1, 1) -> (B, embed_dim)
        aggregated = aggregated.flatten(2).squeeze(-1)

        # Combine with zero-initialized MLP
        combined = resized_embedding + self.zero_mlps[scale_idx - 1](aggregated)

        return combined

    def forward(
        self,
        patches: List[torch.Tensor],
        scales: List[int]
    ) -> torch.Tensor:
        """
        Embed a batch of patches with potentially different scales.

        Args:
            patches: List of patches, each of shape (C, H, W)
            scales: List of scale indices

        Returns:
            Embeddings of shape (num_patches, embed_dim)
        """
        embeddings = []

        for patch, scale in zip(patches, scales):
            emb = self.embed_multi_scale_patch(patch, scale)
            embeddings.append(emb)

        # Stack all embeddings
        embeddings = torch.cat(embeddings, dim=0)  # (num_patches, embed_dim)

        return embeddings
