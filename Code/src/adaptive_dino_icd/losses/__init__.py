"""
Loss functions for Asymmetric Similarity Learning (ASL).

Implements ASL loss for both self-supervised (DISC) and supervised (NDEC) streams.
"""

from adaptive_dino_icd.losses.metric_losses import MetricLoss, ContrastiveLoss, NormRatioLoss, TripletLoss
from adaptive_dino_icd.losses.asl_loss import ASLLossModule, ASLLossWithLocalFeatures

__all__ = [
    "MetricLoss",
    "ContrastiveLoss",
    "NormRatioLoss",
    "TripletLoss",
    "ASLLossModule",
    "ASLLossWithLocalFeatures",
]
