"""
Adaptive Patch Tokenization (APT) module.

Implements entropy-based adaptive patch sizing and token aggregation
for variable-resolution feature extraction.
"""

from adaptive_dino_icd.apt.entropy_scorer import EntropyScorer
from adaptive_dino_icd.apt.patch_selector import PatchSelector, PatchRegion
from adaptive_dino_icd.apt.patch_aggregator import PatchAggregator, ZeroMLP
from adaptive_dino_icd.apt.sequence_packer import SequencePacker

__all__ = [
    "EntropyScorer",
    "PatchSelector",
    "PatchRegion",
    "PatchAggregator",
    "ZeroMLP",
    "SequencePacker",
]
