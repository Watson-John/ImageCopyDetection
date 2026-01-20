"""
Asymmetric Similarity Learning (ASL) Loss Module.

Implements the complete ASL loss combining:
- Norm ratio loss for direction-aware similarity
- Metric loss for embedding quality

Handles both DISC (self-supervised) and NDEC (supervised) streams.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_dino_icd.losses.metric_losses import MetricLoss, NormRatioLoss

logger = logging.getLogger(__name__)


class ASLLossModule(nn.Module):
    """
    Asymmetric Similarity Learning Loss Module.

    Computes ASL loss for mixed DISC/NDEC batches:

    For positive pairs:
        L = exp(1 - R(ref -> query)) + lambda * L_mtr
        where R = ||f(ref)|| / ||f(query)||

    For NDEC negatives:
        - Default: treated as negatives in L_mtr (push apart)
        - If similar_pair_for_metric=True: treated as positive for L_mtr
        - Norm ratio only applied when direction annotation indicates asymmetry

    Phase 1: Uses global features only (CLS token or mean-pooled patches).
    Local features are returned by backbone but NOT used in loss.

    Args:
        lambda_mtr: Weight for metric loss term (default: 0.5)
        temperature: Temperature for metric loss (default: 0.07)
        margin: Margin for metric loss (default: 0.2)
        normalize_features: Whether to L2-normalize features (default: True)
        eps: Small constant for numerical stability

    Example:
        loss_module = ASLLossModule(lambda_mtr=0.5)

        outputs = model(images)  # Contains global_feats, local_feats, etc.
        loss, info = loss_module(outputs_ref, outputs_query, batch)
    """

    def __init__(
        self,
        lambda_mtr: float = 0.5,
        temperature: float = 0.07,
        margin: float = 0.2,
        normalize_features: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_mtr = lambda_mtr
        self.temperature = temperature
        self.margin = margin
        self.normalize_features = normalize_features
        self.eps = eps

        # Sub-losses
        self.metric_loss = MetricLoss(
            margin=margin,
            temperature=temperature,
            reduction="none",  # We handle reduction ourselves
        )

        self.norm_ratio_loss = NormRatioLoss(
            reduction="none",
            eps=eps,
        )

        logger.info(
            f"ASLLossModule initialized: lambda_mtr={lambda_mtr}, "
            f"temperature={temperature}, margin={margin}"
        )

    def forward(
        self,
        feat_ref: torch.Tensor,
        feat_query: torch.Tensor,
        is_copy: torch.Tensor,
        stream: List[str],
        direction: List[str],
        similar_pair_for_metric: Optional[List[bool]] = None,
        labels: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ASL loss for a batch.

        Args:
            feat_ref: Reference features (B, D) - global features
            feat_query: Query features (B, D) - global features
            is_copy: Boolean tensor (B,) indicating positive pairs
            stream: List of stream labels ("disc" or "ndec")
            direction: List of direction strings (e.g., "full->crop", "a->b")
            similar_pair_for_metric: Optional list of bools for NDEC pairs
            labels: Optional list of labels ("pos" or "neg")

        Returns:
            (loss, info_dict) - total loss and logging info
        """
        B = feat_ref.shape[0]
        device = feat_ref.device

        # Ensure features are on same device
        feat_query = feat_query.to(device)
        is_copy = is_copy.to(device)

        # Optionally normalize features for metric loss
        # (keep unnormalized for norm ratio loss)
        if self.normalize_features:
            feat_ref_norm = F.normalize(feat_ref, p=2, dim=-1)
            feat_query_norm = F.normalize(feat_query, p=2, dim=-1)
        else:
            feat_ref_norm = feat_ref
            feat_query_norm = feat_query

        # Create masks for different streams
        disc_mask = torch.tensor([s == "disc" for s in stream], device=device)
        ndec_mask = torch.tensor([s == "ndec" for s in stream], device=device)

        # Create mask for applying norm ratio loss
        # Apply to: all DISC samples (positive) + NDEC positives with direction
        apply_norm_ratio = self._get_norm_ratio_mask(
            is_copy, stream, direction, labels, device
        )

        # Compute norm ratio loss
        norm_loss, norm_info = self.norm_ratio_loss(
            feat_ref, feat_query, apply_mask=apply_norm_ratio
        )

        # Reduce norm_loss if it's per-sample (reduction='none')
        if isinstance(norm_loss, torch.Tensor) and norm_loss.dim() > 0:
            n_valid = apply_norm_ratio.sum().clamp(min=1)
            norm_loss = norm_loss.sum() / n_valid

        # Create similar_for_metric tensor
        if similar_pair_for_metric is not None:
            similar_tensor = torch.tensor(similar_pair_for_metric, device=device)
        else:
            similar_tensor = torch.zeros(B, dtype=torch.bool, device=device)

        # Compute metric loss
        mtr_loss, mtr_info = self.metric_loss(
            feat_ref_norm, feat_query_norm, is_copy, similar_tensor
        )

        # Combine per-sample losses
        # norm_loss is scalar, mtr_loss is per-sample if reduction='none'
        if isinstance(mtr_loss, torch.Tensor) and mtr_loss.dim() > 0:
            mtr_loss = mtr_loss.mean()

        total_loss = norm_loss + self.lambda_mtr * mtr_loss

        # Build info dict
        info = {
            "total_loss": total_loss.item(),
            "norm_ratio_loss": norm_info["norm_ratio_loss"],
            "metric_loss": mtr_info["metric_loss"],
            "avg_ratio": norm_info["avg_ratio"],
            "avg_ref_norm": norm_info["avg_ref_norm"],
            "avg_query_norm": norm_info["avg_query_norm"],
            "avg_pos_sim": mtr_info["avg_pos_sim"],
            "avg_neg_sim": mtr_info["avg_neg_sim"],
            "disc_count": disc_mask.sum().item(),
            "ndec_count": ndec_mask.sum().item(),
            "pos_count": is_copy.sum().item(),
            "neg_count": (~is_copy).sum().item(),
        }

        return total_loss, info

    def _get_norm_ratio_mask(
        self,
        is_copy: torch.Tensor,
        stream: List[str],
        direction: List[str],
        labels: Optional[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create mask for applying norm ratio loss.

        Norm ratio loss is applied to:
        1. All DISC samples (always positive with direction full->crop)
        2. NDEC positive pairs where direction indicates asymmetry

        Args:
            is_copy: Boolean tensor of positive labels
            stream: List of stream labels
            direction: List of direction strings
            labels: Optional list of labels
            device: Target device

        Returns:
            Boolean mask (B,)
        """
        B = is_copy.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            if stream[i] == "disc":
                # Always apply to DISC (full->crop is asymmetric)
                mask[i] = True
            elif stream[i] == "ndec":
                # Apply to NDEC only if:
                # 1. It's a positive pair (is_copy)
                # 2. Direction indicates asymmetry
                is_positive = is_copy[i].item() if isinstance(is_copy[i], torch.Tensor) else is_copy[i]
                if is_positive:
                    # Check if direction has clear asymmetry
                    dir_str = direction[i]
                    if "->" in dir_str:  # e.g., "a->b" indicates a is original
                        mask[i] = True

        return mask

    def compute_disc_loss(
        self,
        feat_full: torch.Tensor,
        feat_crop: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for DISC-only batch (self-supervised).

        All pairs are positive (full, crop) with direction full->crop.

        Args:
            feat_full: Full image features (B, D)
            feat_crop: Crop image features (B, D)

        Returns:
            (loss, info_dict)
        """
        B = feat_full.shape[0]
        device = feat_full.device

        # All are positive with direction full->crop
        is_copy = torch.ones(B, dtype=torch.bool, device=device)
        stream = ["disc"] * B
        direction = ["full->crop"] * B

        return self.forward(
            feat_full, feat_crop, is_copy, stream, direction
        )

    def compute_ndec_loss(
        self,
        feat_ref: torch.Tensor,
        feat_query: torch.Tensor,
        is_copy: torch.Tensor,
        direction: List[str],
        similar_pair_for_metric: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for NDEC-only batch (supervised).

        Args:
            feat_ref: Reference features (B, D) based on direction
            feat_query: Query features (B, D)
            is_copy: Boolean tensor of positive labels
            direction: List of direction strings
            similar_pair_for_metric: Optional list of bools

        Returns:
            (loss, info_dict)
        """
        B = feat_ref.shape[0]
        stream = ["ndec"] * B
        labels = ["pos" if c else "neg" for c in is_copy.tolist()]

        return self.forward(
            feat_ref, feat_query, is_copy, stream, direction,
            similar_pair_for_metric, labels
        )


class ASLLossWithLocalFeatures(ASLLossModule):
    """
    Extended ASL loss that can utilize local features (Phase 2).

    Currently a placeholder that behaves identically to ASLLossModule.
    Will be extended in Phase 2 to incorporate local feature matching.
    """

    def __init__(
        self,
        lambda_mtr: float = 0.5,
        lambda_local: float = 0.0,  # Disabled in Phase 1
        **kwargs,
    ):
        super().__init__(lambda_mtr=lambda_mtr, **kwargs)
        self.lambda_local = lambda_local

        if lambda_local > 0:
            logger.warning(
                "lambda_local > 0 but local feature loss not implemented in Phase 1. "
                "Setting lambda_local=0"
            )
            self.lambda_local = 0.0

    def forward_with_local(
        self,
        global_ref: torch.Tensor,
        global_query: torch.Tensor,
        local_ref: torch.Tensor,
        local_query: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward with both global and local features.

        Phase 1: Only uses global features.
        Phase 2: Will incorporate local feature matching.

        Args:
            global_ref: Global reference features (B, D)
            global_query: Global query features (B, D)
            local_ref: Local reference features (B, N, D) - unused in Phase 1
            local_query: Local query features (B, N, D) - unused in Phase 1
            **kwargs: Additional arguments passed to parent forward

        Returns:
            (loss, info_dict)
        """
        # Phase 1: Only use global features
        return self.forward(global_ref, global_query, **kwargs)
