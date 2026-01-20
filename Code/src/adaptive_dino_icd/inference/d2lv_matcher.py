"""
D2LV-inspired Matcher for copy detection inference.

Implements global-local + local-global scoring interface.
Phase 1: Simple implementation without full retrieval infrastructure.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_dino_icd.inference.scoring import (
    global_similarity,
    local_similarity,
    combined_score,
)

logger = logging.getLogger(__name__)


class D2LVMatcher(nn.Module):
    """
    D2LV-inspired matcher for image copy detection.

    Computes similarity scores using both global and local features
    with bidirectional matching (global-local + local-global).

    Phase 1: Simple implementation for evaluation.
    Phase 2: Full retrieval with indexing and reranking.

    Args:
        global_weight: Weight for global similarity (default: 0.5)
        local_weight: Weight for local similarity (default: 0.5)
        local_method: Method for local matching ("max_pool", "avg_pool", "spatial")
        use_bidirectional: Whether to use bidirectional matching
        normalize_features: Whether to L2-normalize features

    Example:
        matcher = D2LVMatcher()

        # Extract features from model
        query_feats = model.extract_features(query_images)
        gallery_feats = model.extract_features(gallery_images)

        # Match
        scores = matcher.match(query_feats, gallery_feats)
    """

    def __init__(
        self,
        global_weight: float = 0.5,
        local_weight: float = 0.5,
        local_method: str = "max_pool",
        use_bidirectional: bool = True,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.local_method = local_method
        self.use_bidirectional = use_bidirectional
        self.normalize_features = normalize_features

        logger.info(
            f"D2LVMatcher initialized: global_weight={global_weight}, "
            f"local_weight={local_weight}, bidirectional={use_bidirectional}"
        )

    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        gallery_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute similarity scores between query and gallery.

        Args:
            query_feats: Dictionary with keys:
                - global_feats: (N_q, D) or (D,)
                - local_feats: (N_q, N_tokens, D) or (N_tokens, D)
                - local_coords: Optional (N_q, N_tokens, 2)
                - attention_mask: Optional (N_q, N_tokens)
            gallery_feats: Same structure as query_feats

        Returns:
            Similarity scores (N_q, N_g) or scalar if single pair
        """
        return self.match(query_feats, gallery_feats)

    def match(
        self,
        query_feats: Dict[str, torch.Tensor],
        gallery_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Match query features against gallery.

        Args:
            query_feats: Query feature dictionary
            gallery_feats: Gallery feature dictionary

        Returns:
            Similarity matrix (N_q, N_g)
        """
        # Extract features
        q_global = query_feats["global_feats"]
        g_global = gallery_feats["global_feats"]

        # Ensure batch dimension
        if q_global.dim() == 1:
            q_global = q_global.unsqueeze(0)
        if g_global.dim() == 1:
            g_global = g_global.unsqueeze(0)

        N_q = q_global.shape[0]
        N_g = g_global.shape[0]

        # Compute global similarity matrix
        global_sim = self._compute_global_similarity(q_global, g_global)

        # Compute local similarity if available
        if "local_feats" in query_feats and "local_feats" in gallery_feats:
            local_sim = self._compute_local_similarity(
                query_feats, gallery_feats, N_q, N_g
            )

            # Combine scores
            scores = combined_score(
                global_sim, local_sim,
                self.global_weight, self.local_weight
            )
        else:
            # Only use global similarity
            scores = global_sim

        return scores

    def _compute_global_similarity(
        self,
        q_global: torch.Tensor,
        g_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global similarity matrix.

        Args:
            q_global: Query global features (N_q, D)
            g_global: Gallery global features (N_g, D)

        Returns:
            Similarity matrix (N_q, N_g)
        """
        if self.normalize_features:
            q_global = F.normalize(q_global, p=2, dim=-1)
            g_global = F.normalize(g_global, p=2, dim=-1)

        # Cosine similarity
        return torch.mm(q_global, g_global.t())

    def _compute_local_similarity(
        self,
        query_feats: Dict[str, torch.Tensor],
        gallery_feats: Dict[str, torch.Tensor],
        N_q: int,
        N_g: int,
    ) -> torch.Tensor:
        """
        Compute local similarity matrix.

        Args:
            query_feats: Query feature dictionary with local_feats
            gallery_feats: Gallery feature dictionary with local_feats
            N_q: Number of query images
            N_g: Number of gallery images

        Returns:
            Local similarity matrix (N_q, N_g)
        """
        q_local = query_feats["local_feats"]
        g_local = gallery_feats["local_feats"]

        q_coords = query_feats.get("local_coords")
        g_coords = gallery_feats.get("local_coords")

        q_mask = query_feats.get("attention_mask")
        g_mask = gallery_feats.get("attention_mask")

        device = q_local.device
        local_sim = torch.zeros(N_q, N_g, device=device)

        # Compute pairwise local similarity
        for i in range(N_q):
            for j in range(N_g):
                qi_local = q_local[i] if q_local.dim() == 3 else q_local
                gj_local = g_local[j] if g_local.dim() == 3 else g_local

                qi_coords = q_coords[i] if q_coords is not None and q_coords.dim() == 3 else q_coords
                gj_coords = g_coords[j] if g_coords is not None and g_coords.dim() == 3 else g_coords

                qi_mask = q_mask[i] if q_mask is not None and q_mask.dim() == 2 else q_mask
                gj_mask = g_mask[j] if g_mask is not None and g_mask.dim() == 2 else g_mask

                # Forward direction: query->gallery
                sim_fwd = local_similarity(
                    qi_local, gj_local,
                    qi_coords, gj_coords,
                    qi_mask, gj_mask,
                    method=self.local_method,
                    normalize=self.normalize_features,
                )

                if self.use_bidirectional:
                    # Backward direction: gallery->query
                    sim_bwd = local_similarity(
                        gj_local, qi_local,
                        gj_coords, qi_coords,
                        gj_mask, qi_mask,
                        method=self.local_method,
                        normalize=self.normalize_features,
                    )

                    # Average bidirectional
                    local_sim[i, j] = (sim_fwd + sim_bwd) / 2
                else:
                    local_sim[i, j] = sim_fwd

        return local_sim

    def match_single(
        self,
        query_feats: Dict[str, torch.Tensor],
        gallery_feats: Dict[str, torch.Tensor],
    ) -> float:
        """
        Match a single query against single gallery image.

        Args:
            query_feats: Single query features (no batch dim)
            gallery_feats: Single gallery features (no batch dim)

        Returns:
            Similarity score (scalar)
        """
        scores = self.match(query_feats, gallery_feats)
        return scores.item()

    def rerank(
        self,
        query_feats: Dict[str, torch.Tensor],
        gallery_feats: Dict[str, torch.Tensor],
        initial_scores: torch.Tensor,
        top_k: int = 100,
    ) -> torch.Tensor:
        """
        Rerank initial retrieval results using detailed matching.

        Phase 1: Simple reranking using combined scores on top-k.
        Phase 2: More sophisticated reranking with geometric verification.

        Args:
            query_feats: Query features
            gallery_feats: Gallery features
            initial_scores: Initial similarity scores (N_q, N_g)
            top_k: Number of top candidates to rerank per query

        Returns:
            Reranked scores (N_q, N_g)
        """
        N_q, N_g = initial_scores.shape
        device = initial_scores.device

        # Get top-k indices per query
        _, top_indices = initial_scores.topk(min(top_k, N_g), dim=-1)  # (N_q, k)

        # Initialize output with initial scores
        reranked_scores = initial_scores.clone()

        # Recompute scores for top-k candidates
        for i in range(N_q):
            top_k_indices = top_indices[i]

            # Extract features for this query
            q_feats = {
                k: v[i:i+1] if v.dim() > 1 else v
                for k, v in query_feats.items()
            }

            for j_idx, j in enumerate(top_k_indices):
                g_feats = {
                    k: v[j:j+1] if v.dim() > 1 else v
                    for k, v in gallery_feats.items()
                }

                # Compute detailed score
                detailed_score = self.match(q_feats, g_feats)
                reranked_scores[i, j] = detailed_score.item()

        return reranked_scores


class SimpleMatcher(nn.Module):
    """
    Simple global-only matcher for baseline comparison.

    Uses only global features with cosine similarity.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        gallery_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute global similarity matrix."""
        q_global = query_feats["global_feats"]
        g_global = gallery_feats["global_feats"]

        if q_global.dim() == 1:
            q_global = q_global.unsqueeze(0)
        if g_global.dim() == 1:
            g_global = g_global.unsqueeze(0)

        if self.normalize:
            q_global = F.normalize(q_global, p=2, dim=-1)
            g_global = F.normalize(g_global, p=2, dim=-1)

        return torch.mm(q_global, g_global.t())
