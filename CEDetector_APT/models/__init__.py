"""
CEDetector with Adaptive Patch Transformers
Model components
"""

from .cedetector_apt import CEDetectorAPT
from .dinov3_backbone import DiNOv3Backbone
from .apt_patch_selector import APTPatchSelector
from .apt_patch_embedding import APTPatchEmbedding
from .feature_aggregation import CEDFeatureAggregation
from .copy_edit_classifier import CopyEditClassifier

__all__ = [
    'CEDetectorAPT',
    'DiNOv3Backbone',
    'APTPatchSelector',
    'APTPatchEmbedding',
    'CEDFeatureAggregation',
    'CopyEditClassifier',
]
