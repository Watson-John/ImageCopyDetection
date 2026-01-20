"""
Inference module for D2LV-inspired matching.

Provides global-local scoring interfaces for copy detection inference.
Phase 1: Simple implementation without full retrieval infrastructure.
"""

from adaptive_dino_icd.inference.scoring import (
    global_similarity,
    local_similarity,
    combined_score,
    compute_similarity_matrix,
)
from adaptive_dino_icd.inference.d2lv_matcher import D2LVMatcher

__all__ = [
    "global_similarity",
    "local_similarity",
    "combined_score",
    "compute_similarity_matrix",
    "D2LVMatcher",
]
