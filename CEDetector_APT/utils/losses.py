"""
Loss Functions for CED Training
Based on CED paper: SimCLR, Multi-Similarity Loss, BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    """
    SimCLR contrastive loss with NT-Xent.
    Maximizes agreement between positive samples.
    """

    def __init__(self, temperature: float = 0.1):  # Increased from 0.025 for stability
        super().__init__()
        self.temperature = max(temperature, 0.05)  # Ensure minimum temperature for stability

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Projected features from view i (B, D)
            z_j: Projected features from view j (B, D)

        Returns:
            Loss value
        """
        B = z_i.shape[0]

        # Normalize features
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Clamp for numerical stability (prevent extreme values)
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)

        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.arange(2 * B, device=z.device)
        labels = (labels + B) % (2 * B)

        # Mask out self-similarities
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        # Use -1e4 for masked values
        sim_matrix = sim_matrix.masked_fill(mask, -50.0)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class KLDivergenceLoss(nn.Module):
    """
    Kozachenko-Leonenko differential entropy estimator.
    Regularizes to increase distance between positive and negative samples.
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Feature embeddings (B, D)

        Returns:
            KL divergence loss
        """
        B = embeddings.shape[0]

        # Compute pairwise L2 distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Mask out self-distances
        mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        dist_matrix = dist_matrix.masked_fill(mask, float('inf'))

        # Get minimum distance for each sample
        min_dists, _ = dist_matrix.min(dim=1)

        # Compute log of minimum distances
        log_min_dists = torch.log(min_dists + 1e-8)

        # Average over batch
        loss = log_min_dists.mean()

        return loss


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss for deep metric learning.
    Learns embedding space where similar samples are close.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 20.0,  # Reduced from 50.0 for numerical stability
        margin: float = 0.1,  # Reduced from 1.0 for stability
        epsilon: float = 0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.epsilon = epsilon

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Feature embeddings (B, D)
            labels: Binary labels (B,) - 1 for positives, 0 for negatives

        Returns:
            MSL loss value
        """
        B = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Create positive and negative masks
        # Convert labels to long for comparison (handle float labels)
        labels_int = labels.long() if labels.dtype == torch.float32 else labels
        labels_int = labels_int.unsqueeze(0)
        pos_mask = (labels_int == labels_int.t()) & (labels_int == 1)
        neg_mask = (labels_int != labels_int.t())

        # Mask out self-similarities
        self_mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask

        # Positive loss term
        pos_sim = sim_matrix - self.margin
        if pos_mask.any():
            pos_sim_values = pos_sim[pos_mask]
            # Clamp to prevent overflow
            pos_sim_values = torch.clamp(pos_sim_values, min=-10, max=10)
            pos_loss = (1.0 / self.alpha) * torch.logsumexp(
                self.alpha * pos_sim_values,
                dim=0
            )
        else:
            pos_loss = torch.tensor(0.0, device=embeddings.device)

        # Negative loss term
        neg_sim = self.margin - sim_matrix
        if neg_mask.any():
            neg_sim_values = neg_sim[neg_mask]
            # Clamp to prevent overflow
            neg_sim_values = torch.clamp(neg_sim_values, min=-10, max=10)
            neg_loss = (1.0 / self.beta) * torch.logsumexp(
                self.beta * neg_sim_values,
                dim=0
            )
        else:
            neg_loss = torch.tensor(0.0, device=embeddings.device)

        # Total loss
        loss = pos_loss + neg_loss

        # Safety check for NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss


class CEDLoss(nn.Module):
    """
    Combined loss function for CED training.
    L = L_contrast + L_MSL + L_BCE
    """

    def __init__(
        self,
        temperature: float = 0.1,  # Increased from 0.025 for stability
        entropy_weight: float = 0.5,
        msl_alpha: float = 2.0,
        msl_beta: float = 20.0,  # Reduced from 50.0 for stability
        msl_margin: float = 0.1,  # Reduced from 1.0 for stability
    ):
        super().__init__()
        self.simclr_loss = SimCLRLoss(temperature=temperature)
        self.kl_loss = KLDivergenceLoss()
        self.msl_loss = MultiSimilarityLoss(
            alpha=msl_alpha,
            beta=msl_beta,
            margin=msl_margin
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.entropy_weight = entropy_weight

    def forward(
        self,
        cls_tokens_i: torch.Tensor,
        cls_tokens_j: torch.Tensor,
        patch_embeddings: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.

        Args:
            cls_tokens_i: CLS tokens from view i (B, D)
            cls_tokens_j: CLS tokens from view j (B, D)
            patch_embeddings: Patch token embeddings (B, D)
            labels: Binary labels (B,)
            logits: Classification logits (B, 1)

        Returns:
            Dictionary with individual and total losses
        """
        # Contrastive loss (SimCLR + KL divergence)
        simclr_loss = self.simclr_loss(cls_tokens_i, cls_tokens_j)
        kl_loss = self.kl_loss(cls_tokens_i)
        contrast_loss = simclr_loss + self.entropy_weight * kl_loss

        # Multi-similarity loss on patch embeddings
        msl_loss = self.msl_loss(patch_embeddings, labels)

        # Binary cross-entropy loss for classification
        # Ensure logits and labels have compatible shapes
        logits_bce = logits.unsqueeze(1) if logits.dim() == 1 else logits
        labels_bce = labels.unsqueeze(1) if labels.dim() == 1 else labels

        bce_loss = self.bce_loss(logits_bce, labels_bce.float())

        # Total loss
        total_loss = contrast_loss + msl_loss + bce_loss

        return {
            'total_loss': total_loss,
            'contrast_loss': contrast_loss,
            'simclr_loss': simclr_loss,
            'kl_loss': kl_loss,
            'msl_loss': msl_loss,
            'bce_loss': bce_loss,
        }
