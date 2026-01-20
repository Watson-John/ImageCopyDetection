"""
Backbone module for DINOv3 integration with adaptive tokenization.

Provides wrapper classes for loading and using DINOv3 from HuggingFace,
with RoPE position adaptation for variable-sized patches.
"""

from adaptive_dino_icd.backbone.dinov3_hf_wrapper import Dinov3HFWrapper, ModelNotFoundError
from adaptive_dino_icd.backbone.rope_adapter import RoPEAdapter
from adaptive_dino_icd.backbone.adaptive_backbone import AdaptiveBackbone

__all__ = [
    "Dinov3HFWrapper",
    "ModelNotFoundError",
    "RoPEAdapter",
    "AdaptiveBackbone",
]
