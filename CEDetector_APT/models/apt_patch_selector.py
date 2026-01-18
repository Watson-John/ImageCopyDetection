"""
Adaptive Patch Transformer (APT) Patch Selection Module
Based on: "Accelerating Vision Transformers with Adaptive Patch Sizes"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class APTPatchSelector:
    """
    Selects patches of varying sizes based on image entropy.
    Uses hierarchical quadtree structure with multiple patch scales.
    """

    def __init__(
        self,
        base_patch_size: int = 16,
        num_scales: int = 3,
        thresholds: List[float] = [5.5, 4.0],
        num_bins: int = 256
    ):
        """
        Args:
            base_patch_size: Smallest patch size (e.g., 16 for 16x16)
            num_scales: Number of patch scales (S in paper)
            thresholds: Entropy thresholds for each scale
            num_bins: Number of bins for entropy calculation
        """
        self.base_patch_size = base_patch_size
        self.num_scales = num_scales
        self.thresholds = thresholds
        self.num_bins = num_bins

    def calculate_entropy(self, patch: torch.Tensor) -> float:
        """
        Calculate entropy of a patch using pixel intensity distribution.

        Args:
            patch: Image patch of shape (C, H, W)

        Returns:
            Entropy value
        """
        # Convert to grayscale if RGB
        if patch.shape[0] == 3:
            # Use luminance formula
            gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
        else:
            gray = patch[0]

        # Normalize to [0, 1]
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Create histogram
        hist = torch.histc(gray.flatten(), bins=self.num_bins, min=0.0, max=1.0)

        # Normalize histogram to get probabilities
        hist = hist / (hist.sum() + 1e-8)

        # Calculate entropy: H = -sum(p * log2(p))
        # Filter out zero probabilities
        hist = hist[hist > 0]
        entropy = -(hist * torch.log2(hist)).sum()

        return entropy.item()

    def get_adaptive_patches(
        self,
        image: torch.Tensor,
        return_positions: bool = True
    ) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, int]]]:
        """
        Extract patches of varying sizes based on entropy.

        Args:
            image: Input image of shape (C, H, W)
            return_positions: Whether to return patch positions

        Returns:
            patches: List of patches
            scales: List of scale indices for each patch
            positions: List of (row, col) positions for each patch
        """
        C, H, W = image.shape
        patches = []
        scales = []
        positions = []

        # Create a mask to track which regions have been assigned
        assigned = torch.zeros(H // self.base_patch_size, W // self.base_patch_size, dtype=torch.bool)

        # Process from coarsest to finest scale
        for scale_idx in range(self.num_scales - 1, -1, -1):
            patch_size = self.base_patch_size * (2 ** scale_idx)
            stride = patch_size

            # Get threshold for this scale
            threshold = self.thresholds[min(scale_idx, len(self.thresholds) - 1)]

            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    # Check if this region has already been assigned at a coarser scale
                    base_i = i // self.base_patch_size
                    base_j = j // self.base_patch_size
                    base_size = patch_size // self.base_patch_size

                    if assigned[base_i:base_i+base_size, base_j:base_j+base_size].any():
                        continue

                    # Extract patch
                    patch = image[:, i:i+patch_size, j:j+patch_size]

                    # Calculate entropy
                    entropy = self.calculate_entropy(patch)

                    # If entropy is below threshold and not at finest scale, use larger patch
                    if entropy < threshold and scale_idx < self.num_scales - 1:
                        # Mark this region as assigned
                        assigned[base_i:base_i+base_size, base_j:base_j+base_size] = True
                        patches.append(patch)
                        scales.append(scale_idx)
                        if return_positions:
                            positions.append((i, j))
                    # If at finest scale, use this patch regardless
                    elif scale_idx == 0:
                        assigned[base_i:base_i+base_size, base_j:base_j+base_size] = True
                        patches.append(patch)
                        scales.append(scale_idx)
                        if return_positions:
                            positions.append((i, j))

        return patches, scales, positions

    def extract_query_patches(
        self,
        image: torch.Tensor,
        num_patches: int = 6,
        overlap_ratio: float = 0.2
    ) -> List[torch.Tensor]:
        """
        Extract fixed number of overlapping patches for query image.
        Used in CED paper - 6 patches per query image.

        Args:
            image: Input image of shape (C, H, W)
            num_patches: Number of patches to extract (default: 6)
            overlap_ratio: Overlap between patches

        Returns:
            List of query patches
        """
        C, H, W = image.shape

        # Calculate patch arrangement (e.g., 2x3 for 6 patches)
        if num_patches == 6:
            rows, cols = 2, 3
        elif num_patches == 4:
            rows, cols = 2, 2
        else:
            rows = cols = int(np.sqrt(num_patches))

        patch_h = int(H * (1 + overlap_ratio) / rows)
        patch_w = int(W * (1 + overlap_ratio) / cols)

        stride_h = int(H * (1 - overlap_ratio) / (rows - 1)) if rows > 1 else 0
        stride_w = int(W * (1 - overlap_ratio) / (cols - 1)) if cols > 1 else 0

        patches = []
        for i in range(rows):
            for j in range(cols):
                start_h = min(i * stride_h, H - patch_h)
                start_w = min(j * stride_w, W - patch_w)

                patch = image[:, start_h:start_h+patch_h, start_w:start_w+patch_w]
                patches.append(patch)

        return patches[:num_patches]
