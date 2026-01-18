"""Utilities for CEDetector with APT"""

from .ced_augmentations import CEDAugmentationPipeline, build_ced_transforms
from .losses import CEDLoss, SimCLRLoss, MultiSimilarityLoss
from .metrics import MetricsTracker, compute_micro_ap, compute_recall_at_precision

# Alias for compatibility
CEDAugmentations = CEDAugmentationPipeline

__all__ = [
    'CEDAugmentations',
    'CEDAugmentationPipeline',
    'build_ced_transforms',
    'CEDLoss',
    'SimCLRLoss',
    'MultiSimilarityLoss',
    'MetricsTracker',
    'compute_micro_ap',
    'compute_recall_at_precision',
]
