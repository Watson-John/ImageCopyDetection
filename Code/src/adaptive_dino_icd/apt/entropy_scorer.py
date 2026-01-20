"""
Entropy Scorer for Adaptive Patch Tokenization.

Computes local entropy at multiple scales to guide adaptive patch sizing.
High entropy regions (texture, edges) get smaller patches;
low entropy regions (uniform areas) get larger patches.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyScorer(nn.Module):
    """
    Compute local entropy maps at multiple scales.

    Entropy is computed per cell using histogram-based estimation on grayscale values.
    Higher entropy indicates more information content (texture, edges, detail).

    Args:
        scales: List of patch sizes for entropy computation (e.g., [8, 16, 32])
        num_bins: Number of histogram bins for entropy estimation
        normalize: Whether to normalize entropy to [0, 1] range
        eps: Small constant for numerical stability

    Example:
        scorer = EntropyScorer(scales=[8, 16, 32])
        images = torch.randn(4, 3, 224, 224)
        entropy_maps = scorer(images)
        # entropy_maps[8].shape == (4, 28, 28) for 224/8=28 cells
    """

    def __init__(
        self,
        scales: List[int] = [8, 16, 32],
        num_bins: int = 32,
        normalize: bool = True,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.scales = sorted(scales)
        self.num_bins = num_bins
        self.normalize = normalize
        self.eps = eps

        # Maximum possible entropy for normalization
        self.max_entropy = torch.log(torch.tensor(num_bins, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute entropy maps at multiple scales.

        Args:
            images: Input images (B, C, H, W) in range [0, 1] or [-1, 1]

        Returns:
            Dictionary mapping scale -> entropy map (B, H//scale, W//scale)
        """
        B, C, H, W = images.shape

        # Convert to grayscale if needed
        if C == 3:
            # Standard RGB to grayscale conversion
            gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        elif C == 1:
            gray = images[:, 0]
        else:
            gray = images.mean(dim=1)

        # Normalize to [0, 1] if needed
        if gray.min() < 0:
            gray = (gray + 1) / 2
        gray = gray.clamp(0, 1)

        entropy_maps = {}
        for scale in self.scales:
            if H % scale != 0 or W % scale != 0:
                # Pad to make divisible
                pad_h = (scale - H % scale) % scale
                pad_w = (scale - W % scale) % scale
                gray_padded = F.pad(gray, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                gray_padded = gray
                pad_h, pad_w = 0, 0

            H_pad, W_pad = gray_padded.shape[1], gray_padded.shape[2]
            n_cells_h = H_pad // scale
            n_cells_w = W_pad // scale

            # Reshape to cells
            # (B, H, W) -> (B, n_cells_h, scale, n_cells_w, scale)
            cells = gray_padded.view(B, n_cells_h, scale, n_cells_w, scale)
            # -> (B, n_cells_h, n_cells_w, scale, scale)
            cells = cells.permute(0, 1, 3, 2, 4).contiguous()
            # -> (B, n_cells_h, n_cells_w, scale*scale)
            cells = cells.view(B, n_cells_h, n_cells_w, scale * scale)

            # Compute entropy per cell
            entropy = self._compute_histogram_entropy(cells)

            entropy_maps[scale] = entropy

        return entropy_maps

    def _compute_histogram_entropy(self, cells: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy from histograms of cell values.

        Args:
            cells: Cell values (B, H_cells, W_cells, N_pixels)

        Returns:
            Entropy per cell (B, H_cells, W_cells)
        """
        B, H_cells, W_cells, N_pixels = cells.shape
        device = cells.device

        # Quantize values to bins
        bin_indices = (cells * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

        # Compute histogram using one-hot encoding (differentiable approximation)
        # Shape: (B, H_cells, W_cells, N_pixels, num_bins)
        one_hot = F.one_hot(bin_indices, self.num_bins).float()

        # Sum over pixels to get histogram: (B, H_cells, W_cells, num_bins)
        histogram = one_hot.sum(dim=3)

        # Normalize to probability distribution
        probabilities = histogram / (N_pixels + self.eps)

        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(probabilities + self.eps)
        entropy = -(probabilities * log_probs).sum(dim=-1)

        if self.normalize:
            entropy = entropy / (self.max_entropy.to(device) + self.eps)
            entropy = entropy.clamp(0, 1)

        return entropy

    def get_combined_entropy(
        self, entropy_maps: Dict[int, torch.Tensor], weights: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """
        Combine entropy maps from multiple scales into a single map.

        Args:
            entropy_maps: Dictionary of entropy maps from forward()
            weights: Optional weights per scale (default: equal weights)

        Returns:
            Combined entropy map at the finest scale resolution
        """
        if weights is None:
            weights = {s: 1.0 / len(self.scales) for s in self.scales}

        # Use finest scale as target resolution
        finest_scale = self.scales[0]
        target_size = entropy_maps[finest_scale].shape[-2:]

        combined = None
        for scale, entropy in entropy_maps.items():
            if entropy.shape[-2:] != target_size:
                # Upsample to finest resolution
                entropy = F.interpolate(
                    entropy.unsqueeze(1),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            weighted = entropy * weights.get(scale, 1.0 / len(self.scales))

            if combined is None:
                combined = weighted
            else:
                combined = combined + weighted

        return combined
