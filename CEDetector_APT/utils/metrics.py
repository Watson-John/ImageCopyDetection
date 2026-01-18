"""
Evaluation Metrics for Copy-Edit Detection
Based on CED paper: µAP (micro Average Precision) and R@P90
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from typing import Tuple, Dict


def compute_micro_ap(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute micro Average Precision (µAP).
    This is the area under the precision-recall curve.

    Args:
        predictions: Predicted scores (N,)
        labels: Ground truth binary labels (N,)

    Returns:
        µAP score
    """
    return average_precision_score(labels, predictions)


def compute_recall_at_precision(
    predictions: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.9
) -> float:
    """
    Compute Recall at target Precision (R@P90).

    Args:
        predictions: Predicted scores (N,)
        labels: Ground truth binary labels (N,)
        target_precision: Target precision threshold (default: 0.9)

    Returns:
        Recall value at target precision
    """
    precision, recall, thresholds = precision_recall_curve(labels, predictions)

    # Find recall at target precision
    valid_idx = np.where(precision >= target_precision)[0]

    if len(valid_idx) == 0:
        return 0.0

    # Return maximum recall at target precision
    return recall[valid_idx].max()


def evaluate_retrieval(
    query_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    ground_truth: Dict[str, str],
    query_ids: list,
    reference_ids: list,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.

    Args:
        query_descriptors: Query descriptors (N_q, D)
        reference_descriptors: Reference descriptors (N_r, D)
        ground_truth: Dictionary mapping query_id -> reference_id
        query_ids: List of query IDs
        reference_ids: List of reference IDs
        k: Number of top results to consider

    Returns:
        Dictionary with evaluation metrics
    """
    # Normalize descriptors
    query_descriptors = torch.nn.functional.normalize(query_descriptors, dim=1)
    reference_descriptors = torch.nn.functional.normalize(reference_descriptors, dim=1)

    # Compute similarity matrix
    similarities = torch.mm(query_descriptors, reference_descriptors.t())

    # Get top-k retrievals for each query
    top_k_values, top_k_indices = torch.topk(similarities, k, dim=1)

    # Compute metrics
    correct_at_k = []
    mrr_scores = []

    for i, query_id in enumerate(query_ids):
        if query_id not in ground_truth:
            continue

        true_ref_id = ground_truth[query_id]

        # Get retrieved reference IDs
        retrieved_ref_ids = [reference_ids[idx] for idx in top_k_indices[i].cpu().numpy()]

        # Check if true reference is in top-k
        if true_ref_id in retrieved_ref_ids:
            correct_at_k.append(1.0)

            # Compute reciprocal rank
            rank = retrieved_ref_ids.index(true_ref_id) + 1
            mrr_scores.append(1.0 / rank)
        else:
            correct_at_k.append(0.0)
            mrr_scores.append(0.0)

    metrics = {
        f'recall@{k}': np.mean(correct_at_k) if correct_at_k else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
    }

    return metrics


def evaluate_classification(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate classification performance.

    Args:
        predictions: Predicted scores (N,)
        labels: Ground truth binary labels (N,)

    Returns:
        Dictionary with evaluation metrics
    """
    # Compute µAP
    mu_ap = compute_micro_ap(predictions, labels)

    # Compute R@P90
    r_at_p90 = compute_recall_at_precision(predictions, labels, target_precision=0.9)

    # Compute accuracy at threshold 0.5
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = (binary_preds == labels).mean()

    metrics = {
        'mu_ap': mu_ap,
        'r_at_p90': r_at_p90,
        'accuracy': accuracy,
    }

    return metrics


class MetricsTracker:
    """Track metrics during training and evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.labels = []
        self.losses = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: float = None
    ):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted scores
            labels: Ground truth labels
            loss: Optional loss value
        """
        self.predictions.extend(predictions.detach().cpu().numpy().tolist())
        self.labels.extend(labels.detach().cpu().numpy().tolist())

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary with all metrics
        """
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)

        metrics = evaluate_classification(predictions, labels)

        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)

        return metrics

    def print_metrics(self, prefix: str = ""):
        """Print metrics in a readable format."""
        metrics = self.compute()

        print(f"\n{prefix} Metrics:")
        print(f"  µAP: {metrics['mu_ap']:.4f}")
        print(f"  R@P90: {metrics['r_at_p90']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        if 'avg_loss' in metrics:
            print(f"  Avg Loss: {metrics['avg_loss']:.4f}")
