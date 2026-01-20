"""
Patch Selector for Adaptive Patch Tokenization.

Implements quadtree-style adaptive patch selection based on entropy thresholds.
High-entropy regions are split into smaller patches, low-entropy regions use larger patches.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PatchRegion:
    """
    Represents a selected patch region in an image.

    Attributes:
        x: Top-left x coordinate (in pixels)
        y: Top-left y coordinate (in pixels)
        w: Width (in pixels)
        h: Height (in pixels)
        level: Quadtree level (0 = largest, higher = smaller patches)
        entropy: Average entropy of this region
    """

    x: int
    y: int
    w: int
    h: int
    level: int = 0
    entropy: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        """Return center coordinates."""
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> int:
        """Return area in pixels."""
        return self.w * self.h

    def to_normalized_coords(self, img_h: int, img_w: int) -> Tuple[float, float]:
        """
        Convert center to normalized [-1, 1] coordinates.

        Args:
            img_h: Image height
            img_w: Image width

        Returns:
            (x_norm, y_norm) in range [-1, 1]
        """
        cx, cy = self.center
        x_norm = (cx / img_w) * 2 - 1
        y_norm = (cy / img_h) * 2 - 1
        return (x_norm, y_norm)


class PatchSelector(nn.Module):
    """
    Quadtree-style adaptive patch selection based on entropy.

    Starting from max_patch_size, recursively splits patches whose entropy
    exceeds the threshold for that level, until min_patch_size is reached.

    Args:
        min_patch_size: Minimum patch size (won't split smaller)
        max_patch_size: Maximum patch size (starting size)
        entropy_thresholds: Dict mapping patch_size -> split threshold
                           Patches with entropy > threshold are split
        default_threshold: Default threshold if size not in entropy_thresholds

    Example:
        selector = PatchSelector(
            min_patch_size=8,
            max_patch_size=64,
            entropy_thresholds={64: 0.3, 32: 0.5, 16: 0.7}
        )
        entropy_maps = entropy_scorer(images)
        patches = selector(entropy_maps, image_size=(224, 224))
    """

    def __init__(
        self,
        min_patch_size: int = 8,
        max_patch_size: int = 64,
        entropy_thresholds: Optional[Dict[int, float]] = None,
        default_threshold: float = 0.5,
    ):
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.entropy_thresholds = entropy_thresholds or {64: 0.3, 32: 0.5, 16: 0.7}
        self.default_threshold = default_threshold

        # Validate sizes are powers of 2
        assert min_patch_size > 0 and (min_patch_size & (min_patch_size - 1)) == 0, \
            "min_patch_size must be power of 2"
        assert max_patch_size > 0 and (max_patch_size & (max_patch_size - 1)) == 0, \
            "max_patch_size must be power of 2"
        assert min_patch_size <= max_patch_size

    def forward(
        self,
        entropy_maps: Dict[int, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> List[List[PatchRegion]]:
        """
        Select patches adaptively based on entropy.

        Args:
            entropy_maps: Dict of entropy maps from EntropyScorer
            image_size: (H, W) of input images

        Returns:
            List of patch lists, one per image in batch
        """
        H, W = image_size

        # Get batch size from any entropy map
        any_map = next(iter(entropy_maps.values()))
        B = any_map.shape[0]

        all_patches = []
        for b in range(B):
            # Extract single-image entropy maps
            single_entropy = {k: v[b] for k, v in entropy_maps.items()}
            patches = self._select_patches_single(single_entropy, H, W)
            all_patches.append(patches)

        return all_patches

    def _select_patches_single(
        self,
        entropy_maps: Dict[int, torch.Tensor],
        img_h: int,
        img_w: int,
    ) -> List[PatchRegion]:
        """Select patches for a single image."""
        patches = []

        # Start with grid of max_patch_size patches
        initial_size = self.max_patch_size
        n_patches_h = img_h // initial_size
        n_patches_w = img_w // initial_size

        # Handle remainder (edge patches)
        remainder_h = img_h % initial_size
        remainder_w = img_w % initial_size

        # Queue of regions to potentially split
        queue = []

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                region = PatchRegion(
                    x=j * initial_size,
                    y=i * initial_size,
                    w=initial_size,
                    h=initial_size,
                    level=0,
                )
                queue.append(region)

        # Handle edge regions if image isn't perfectly divisible
        if remainder_w > 0:
            for i in range(n_patches_h):
                region = PatchRegion(
                    x=n_patches_w * initial_size,
                    y=i * initial_size,
                    w=remainder_w,
                    h=initial_size,
                    level=0,
                )
                queue.append(region)

        if remainder_h > 0:
            for j in range(n_patches_w):
                region = PatchRegion(
                    x=j * initial_size,
                    y=n_patches_h * initial_size,
                    w=initial_size,
                    h=remainder_h,
                    level=0,
                )
                queue.append(region)

        if remainder_w > 0 and remainder_h > 0:
            region = PatchRegion(
                x=n_patches_w * initial_size,
                y=n_patches_h * initial_size,
                w=remainder_w,
                h=remainder_h,
                level=0,
            )
            queue.append(region)

        # Process queue with quadtree splitting
        while queue:
            region = queue.pop(0)

            # Get entropy for this region
            region_entropy = self._get_region_entropy(region, entropy_maps)
            region.entropy = region_entropy

            # Check if we should split
            threshold = self.entropy_thresholds.get(region.w, self.default_threshold)
            should_split = (
                region_entropy > threshold
                and region.w > self.min_patch_size
                and region.h > self.min_patch_size
                and region.w == region.h  # Only split square patches
            )

            if should_split:
                # Split into 4 quadrants
                new_size = region.w // 2
                new_level = region.level + 1

                # Top-left
                queue.append(PatchRegion(
                    x=region.x, y=region.y,
                    w=new_size, h=new_size, level=new_level
                ))
                # Top-right
                queue.append(PatchRegion(
                    x=region.x + new_size, y=region.y,
                    w=new_size, h=new_size, level=new_level
                ))
                # Bottom-left
                queue.append(PatchRegion(
                    x=region.x, y=region.y + new_size,
                    w=new_size, h=new_size, level=new_level
                ))
                # Bottom-right
                queue.append(PatchRegion(
                    x=region.x + new_size, y=region.y + new_size,
                    w=new_size, h=new_size, level=new_level
                ))
            else:
                # Keep this patch
                patches.append(region)

        return patches

    def _get_region_entropy(
        self,
        region: PatchRegion,
        entropy_maps: Dict[int, torch.Tensor],
    ) -> float:
        """Get average entropy for a region using appropriate scale."""
        # Find best matching entropy scale
        best_scale = min(entropy_maps.keys(), key=lambda s: abs(s - region.w))

        entropy_map = entropy_maps[best_scale]
        map_h, map_w = entropy_map.shape

        # Convert region coords to entropy map coords
        scale_h = map_h * best_scale / (map_h * best_scale)  # Image height / scale
        scale_w = map_w * best_scale / (map_w * best_scale)  # Image width / scale

        # Get cell indices
        cell_x_start = int(region.x / best_scale)
        cell_y_start = int(region.y / best_scale)
        cell_x_end = min(int((region.x + region.w) / best_scale), map_w)
        cell_y_end = min(int((region.y + region.h) / best_scale), map_h)

        # Ensure at least one cell
        cell_x_end = max(cell_x_end, cell_x_start + 1)
        cell_y_end = max(cell_y_end, cell_y_start + 1)

        # Extract and average entropy
        region_entropy = entropy_map[
            cell_y_start:cell_y_end, cell_x_start:cell_x_end
        ]

        return region_entropy.mean().item()

    def get_patch_stats(self, patches: List[List[PatchRegion]]) -> Dict[str, float]:
        """
        Compute statistics about patch selection.

        Args:
            patches: Output from forward()

        Returns:
            Dictionary with statistics (avg_count, avg_size, size_distribution, etc.)
        """
        total_patches = sum(len(p) for p in patches)
        if total_patches == 0:
            return {"avg_count": 0, "avg_size": 0}

        all_sizes = []
        all_entropies = []
        for batch_patches in patches:
            for p in batch_patches:
                all_sizes.append(p.w)
                all_entropies.append(p.entropy)

        # Size distribution
        size_counts = {}
        for s in all_sizes:
            size_counts[s] = size_counts.get(s, 0) + 1

        return {
            "avg_count": total_patches / len(patches),
            "avg_size": sum(all_sizes) / len(all_sizes),
            "avg_entropy": sum(all_entropies) / len(all_entropies),
            "size_distribution": size_counts,
            "min_size": min(all_sizes),
            "max_size": max(all_sizes),
        }
