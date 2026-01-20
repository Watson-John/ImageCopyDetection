"""
Scoring functions for image copy detection.

Provides similarity computation at global and local feature levels.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def global_similarity(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute global feature similarity (cosine similarity).

    Args:
        feat_a: Features from image A (B, D) or (D,)
        feat_b: Features from image B (B, D) or (D,)
        normalize: Whether to L2-normalize features first

    Returns:
        Similarity score(s) in [-1, 1] range
        - If inputs are (D,): returns scalar
        - If inputs are (B, D): returns (B,)
    """
    # Handle single vectors
    if feat_a.dim() == 1:
        feat_a = feat_a.unsqueeze(0)
        feat_b = feat_b.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if normalize:
        feat_a = F.normalize(feat_a, p=2, dim=-1)
        feat_b = F.normalize(feat_b, p=2, dim=-1)

    # Cosine similarity
    similarity = (feat_a * feat_b).sum(dim=-1)

    if squeeze_output:
        similarity = similarity.squeeze(0)

    return similarity


def local_similarity(
    local_a: torch.Tensor,
    local_b: torch.Tensor,
    coords_a: Optional[torch.Tensor] = None,
    coords_b: Optional[torch.Tensor] = None,
    mask_a: Optional[torch.Tensor] = None,
    mask_b: Optional[torch.Tensor] = None,
    method: str = "max_pool",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute local feature similarity between two images.

    Args:
        local_a: Local features from image A (N_a, D) or (B, N_a, D)
        local_b: Local features from image B (N_b, D) or (B, N_b, D)
        coords_a: Optional coordinates for A (N_a, 2) or (B, N_a, 2)
        coords_b: Optional coordinates for B (N_b, 2) or (B, N_b, 2)
        mask_a: Optional attention mask for A
        mask_b: Optional attention mask for B
        method: Similarity aggregation method:
            - "max_pool": Maximum over all local matches
            - "avg_pool": Average over all local matches
            - "spatial": Spatial-aware matching using coordinates
        normalize: Whether to L2-normalize features

    Returns:
        Similarity score(s)
    """
    # Handle 2D inputs (single image pair)
    if local_a.dim() == 2:
        local_a = local_a.unsqueeze(0)
        local_b = local_b.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B = local_a.shape[0]

    if normalize:
        local_a = F.normalize(local_a, p=2, dim=-1)
        local_b = F.normalize(local_b, p=2, dim=-1)

    # Compute pairwise similarity matrix
    # (B, N_a, D) x (B, D, N_b) -> (B, N_a, N_b)
    sim_matrix = torch.bmm(local_a, local_b.transpose(-2, -1))

    # Apply masks if provided
    if mask_a is not None and mask_b is not None:
        # Create mask matrix
        mask_matrix = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        sim_matrix = sim_matrix.masked_fill(~mask_matrix, float("-inf"))

    if method == "max_pool":
        # Maximum similarity across all pairs
        # For each query token in A, find best match in B
        max_sim_per_a = sim_matrix.max(dim=-1).values  # (B, N_a)
        if mask_a is not None:
            max_sim_per_a = max_sim_per_a.masked_fill(~mask_a, 0.0)
            similarity = max_sim_per_a.sum(dim=-1) / mask_a.sum(dim=-1).clamp(min=1)
        else:
            similarity = max_sim_per_a.mean(dim=-1)

    elif method == "avg_pool":
        # Average similarity across all pairs
        if mask_a is not None and mask_b is not None:
            valid_count = (mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)).sum(dim=(-1, -2))
            sim_matrix = sim_matrix.masked_fill(~mask_matrix, 0.0)
            similarity = sim_matrix.sum(dim=(-1, -2)) / valid_count.clamp(min=1)
        else:
            similarity = sim_matrix.mean(dim=(-1, -2))

    elif method == "spatial":
        # Spatial-aware matching using coordinates
        if coords_a is None or coords_b is None:
            raise ValueError("Coordinates required for spatial matching")

        similarity = _spatial_similarity(
            local_a, local_b, coords_a, coords_b, sim_matrix, mask_a, mask_b
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    if squeeze_output:
        similarity = similarity.squeeze(0)

    return similarity


def _spatial_similarity(
    local_a: torch.Tensor,
    local_b: torch.Tensor,
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
    sim_matrix: torch.Tensor,
    mask_a: Optional[torch.Tensor] = None,
    mask_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute spatial-aware local similarity.

    Weights matches by spatial proximity of coordinates.
    """
    B, N_a, _ = local_a.shape
    _, N_b, _ = local_b.shape

    # Compute spatial distance matrix
    # coords: (B, N, 2) -> distance: (B, N_a, N_b)
    coords_a = coords_a.unsqueeze(2)  # (B, N_a, 1, 2)
    coords_b = coords_b.unsqueeze(1)  # (B, 1, N_b, 2)

    spatial_dist = torch.norm(coords_a - coords_b, p=2, dim=-1)  # (B, N_a, N_b)

    # Convert distance to weight (closer = higher weight)
    # Using Gaussian weighting
    sigma = 0.5  # Spatial bandwidth
    spatial_weight = torch.exp(-spatial_dist ** 2 / (2 * sigma ** 2))

    # Combine feature similarity with spatial weight
    weighted_sim = sim_matrix * spatial_weight

    # Aggregate
    if mask_a is not None and mask_b is not None:
        mask_matrix = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        weighted_sim = weighted_sim.masked_fill(~mask_matrix, 0.0)
        valid_count = mask_matrix.sum(dim=(-1, -2)).clamp(min=1)
        similarity = weighted_sim.sum(dim=(-1, -2)) / valid_count
    else:
        similarity = weighted_sim.mean(dim=(-1, -2))

    return similarity


def combined_score(
    global_sim: torch.Tensor,
    local_sim: torch.Tensor,
    global_weight: float = 0.5,
    local_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combine global and local similarity scores.

    Args:
        global_sim: Global similarity score(s)
        local_sim: Local similarity score(s)
        global_weight: Weight for global score
        local_weight: Weight for local score

    Returns:
        Combined similarity score(s)
    """
    # Normalize weights
    total_weight = global_weight + local_weight
    global_weight = global_weight / total_weight
    local_weight = local_weight / total_weight

    return global_weight * global_sim + local_weight * local_sim


def compute_similarity_matrix(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise similarity matrix between query and gallery features.

    Args:
        query_feats: Query features (N_q, D)
        gallery_feats: Gallery features (N_g, D)
        normalize: Whether to L2-normalize features

    Returns:
        Similarity matrix (N_q, N_g)
    """
    if normalize:
        query_feats = F.normalize(query_feats, p=2, dim=-1)
        gallery_feats = F.normalize(gallery_feats, p=2, dim=-1)

    # Compute cosine similarity
    similarity = torch.mm(query_feats, gallery_feats.t())

    return similarity


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    ground_truth: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute retrieval metrics from similarity matrix.

    Args:
        similarity_matrix: Similarity matrix (N_q, N_g)
        ground_truth: Ground truth labels (N_q, N_g) or (N_q,) with gallery indices
        k_values: K values for Recall@K computation

    Returns:
        Dictionary of metrics
    """
    N_q = similarity_matrix.shape[0]

    # Get top-k indices per query
    _, top_indices = similarity_matrix.topk(max(k_values), dim=-1)

    # Handle different ground truth formats
    if ground_truth.dim() == 1:
        # Ground truth is index of correct gallery item
        gt_expanded = ground_truth.unsqueeze(-1)
        is_correct = (top_indices == gt_expanded)
    else:
        # Ground truth is binary matrix
        is_correct = ground_truth.gather(1, top_indices) > 0

    metrics = {}

    # Recall@K
    for k in k_values:
        recall_at_k = is_correct[:, :k].any(dim=-1).float().mean().item()
        metrics[f"recall@{k}"] = recall_at_k

    # Mean Average Precision (simplified)
    # For each query, compute AP
    aps = []
    for i in range(N_q):
        correct = is_correct[i].float()
        if correct.sum() == 0:
            aps.append(0.0)
            continue

        # Cumulative correct / position
        cumsum = correct.cumsum(dim=0)
        precision_at_k = cumsum / torch.arange(1, len(correct) + 1, device=correct.device)
        ap = (precision_at_k * correct).sum() / correct.sum()
        aps.append(ap.item())

    metrics["mAP"] = sum(aps) / len(aps) if aps else 0.0

    return metrics
