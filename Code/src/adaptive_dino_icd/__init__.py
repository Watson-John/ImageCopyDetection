"""
Adaptive-DINO Image Copy Detection (ICD)

A modular implementation integrating:
- DINOv3 backbone from HuggingFace
- APT-style adaptive tokenization
- ASL (Asymmetrical-Similarity Learning) loss
- D2LV-inspired inference
"""

__version__ = "0.1.0"

from adaptive_dino_icd.backbone import AdaptiveBackbone, Dinov3HFWrapper, RoPEAdapter
from adaptive_dino_icd.losses import ASLLossModule, MetricLoss
from adaptive_dino_icd.apt import EntropyScorer, PatchSelector, PatchAggregator, SequencePacker

__all__ = [
    "AdaptiveBackbone",
    "Dinov3HFWrapper",
    "RoPEAdapter",
    "ASLLossModule",
    "MetricLoss",
    "EntropyScorer",
    "PatchSelector",
    "PatchAggregator",
    "SequencePacker",
]
