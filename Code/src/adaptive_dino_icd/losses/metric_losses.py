"""
Metric Learning Losses for the L_mtr term in ASL.

Provides contrastive and triplet-style losses for pulling positives together
and pushing negatives apart in the embedding space.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for positive and negative pairs.

    For positive pairs: minimize distance
    For negative pairs: maximize distance (up to margin)

    L_pos = ||f_ref - f_query||^2
    L_neg = max(0, margin - ||f_ref - f_query||)^2

    Args:
        margin: Margin for negative pairs (default: 0.5)
        reduction: Loss reduction ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        feat_ref: torch.Tensor,
        feat_query: torch.Tensor,
        is_positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            feat_ref: Reference features (B, D), assumed normalized
            feat_query: Query features (B, D), assumed normalized
            is_positive: Boolean tensor (B,) indicating positive pairs

        Returns:
            Loss tensor (scalar if reduction != 'none')
        """
        # Compute pairwise distances
        distances = torch.norm(feat_ref - feat_query, p=2, dim=-1)  # (B,)

        # Positive loss: minimize distance
        pos_loss = distances ** 2

        # Negative loss: maximize distance up to margin
        neg_loss = F.relu(self.margin - distances) ** 2

        # Select based on label
        is_positive = is_positive.float()
        loss = is_positive * pos_loss + (1 - is_positive) * neg_loss

        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MetricLoss(nn.Module):
    """
    Combined metric learning loss for ASL.

    Combines:
    1. Cosine similarity loss for positive pairs
    2. Contrastive margin loss for negative pairs

    Args:
        margin: Margin for negative separation (default: 0.2)
        temperature: Temperature for similarity scaling (default: 0.07)
        reduction: Loss reduction method
    """

    def __init__(
        self,
        margin: float = 0.2,
        temperature: float = 0.07,
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        feat_ref: torch.Tensor,
        feat_query: torch.Tensor,
        is_positive: torch.Tensor,
        similar_for_metric: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute metric loss.

        Args:
            feat_ref: Reference features (B, D)
            feat_query: Query features (B, D)
            is_positive: Boolean tensor (B,) for positive/negative pairs
            similar_for_metric: Optional (B,) bool - if True for a negative,
                               treat as positive for metric loss

        Returns:
            (loss, info_dict) - loss scalar and logging info
        """
        # Ensure normalized features
        feat_ref = F.normalize(feat_ref, p=2, dim=-1)
        feat_query = F.normalize(feat_query, p=2, dim=-1)

        # Compute cosine similarity
        similarity = (feat_ref * feat_query).sum(dim=-1)  # (B,)

        # Determine effective labels for metric loss
        if similar_for_metric is not None:
            # Override: treat similar_for_metric=True as positive even if is_positive=False
            effective_positive = is_positive | similar_for_metric
        else:
            effective_positive = is_positive

        effective_positive = effective_positive.float()

        # Positive loss: maximize similarity (minimize 1 - sim)
        pos_loss = 1 - similarity

        # Negative loss: push apart with margin
        # Only penalize if similarity > -margin (i.e., too similar)
        neg_loss = F.relu(similarity + self.margin)

        # Combined loss
        loss = effective_positive * pos_loss + (1 - effective_positive) * neg_loss

        # Temperature scaling
        loss = loss / self.temperature

        # Reduce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        # Logging info
        # Handle both scalar and per-sample loss tensors
        if isinstance(loss, torch.Tensor):
            loss_value = loss.mean().item() if loss.dim() > 0 else loss.item()
        else:
            loss_value = loss

        info = {
            "metric_loss": loss_value,
            "avg_pos_sim": similarity[is_positive].mean().item() if is_positive.any() else 0.0,
            "avg_neg_sim": similarity[~is_positive].mean().item() if (~is_positive).any() else 0.0,
        }

        return loss, info


class TripletLoss(nn.Module):
    """
    Triplet loss for embedding learning.

    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Args:
        margin: Margin between positive and negative distances
        reduction: Loss reduction method
    """

    def __init__(
        self,
        margin: float = 0.3,
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor features (B, D)
            positive: Positive features (B, D)
            negative: Negative features (B, D)

        Returns:
            Loss tensor
        """
        # Distances
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss
        loss = F.relu(d_pos - d_neg + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class NormRatioLoss(nn.Module):
    """
    Norm ratio loss for asymmetric similarity.

    Enforces that ||f(ref)|| > ||f(query)|| for copy detection,
    as the original/source typically has stronger features.

    L = exp(1 - R) where R = ||f(ref)|| / ||f(query)||

    Args:
        reduction: Loss reduction method
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        feat_ref: torch.Tensor,
        feat_query: torch.Tensor,
        apply_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute norm ratio loss.

        Args:
            feat_ref: Reference features (B, D)
            feat_query: Query features (B, D)
            apply_mask: Optional (B,) bool mask - only compute loss where True

        Returns:
            (loss, info_dict)
        """
        # Compute norms
        norm_ref = torch.norm(feat_ref, p=2, dim=-1)  # (B,)
        norm_query = torch.norm(feat_query, p=2, dim=-1)  # (B,)

        # Compute ratio R = ||ref|| / ||query||
        ratio = norm_ref / (norm_query + self.eps)

        # Loss: exp(1 - R)
        # When R > 1 (ref norm > query norm), loss is low
        # When R < 1 (query norm > ref norm), loss is high
        loss = torch.exp(1 - ratio)

        # Apply mask if provided
        if apply_mask is not None:
            loss = loss * apply_mask.float()
            n_valid = apply_mask.sum().clamp(min=1)
        else:
            n_valid = loss.shape[0]

        # Reduce
        if self.reduction == "mean":
            loss = loss.sum() / n_valid
        elif self.reduction == "sum":
            loss = loss.sum()

        # Logging info
        # Handle both scalar and per-sample loss tensors
        if isinstance(loss, torch.Tensor):
            loss_value = loss.mean().item() if loss.dim() > 0 else loss.item()
        else:
            loss_value = loss

        info = {
            "norm_ratio_loss": loss_value,
            "avg_ratio": ratio.mean().item(),
            "avg_ref_norm": norm_ref.mean().item(),
            "avg_query_norm": norm_query.mean().item(),
        }

        return loss, info
