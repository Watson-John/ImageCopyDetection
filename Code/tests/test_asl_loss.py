"""
Tests for ASLLossModule.

Verifies:
- Positive pair with ref->qry gives lower loss when ||ref|| > ||qry||
- Swapping direction changes which ratio is enforced
- Hard negatives increase loss appropriately
- Deterministic outputs with same inputs
"""

import pytest
import torch
import torch.nn.functional as F


class TestASLLoss:
    """Tests for the ASLLossModule class."""

    def test_norm_ratio_direction(self, asl_loss, device):
        """
        Verify positive pair gives lower loss when ||ref|| > ||qry||.

        The ASL loss includes exp(1 - R) where R = ||ref|| / ||qry||.
        When R > 1 (ref norm > query norm), loss should be lower.
        """
        # Create features where ref has larger norm than query
        feat_ref_large = torch.randn(4, 768, device=device)
        feat_ref_large = F.normalize(feat_ref_large, p=2, dim=-1) * 2.0  # Norm = 2

        feat_query_small = torch.randn(4, 768, device=device)
        feat_query_small = F.normalize(feat_query_small, p=2, dim=-1) * 1.0  # Norm = 1

        # Create features where query has larger norm
        feat_ref_small = feat_query_small.clone()  # Norm = 1
        feat_query_large = feat_ref_large.clone()  # Norm = 2

        # All positive pairs
        is_copy = torch.ones(4, dtype=torch.bool, device=device)
        stream = ["disc"] * 4
        direction = ["full->crop"] * 4

        # Loss when ref > query (correct direction)
        loss_correct, info_correct = asl_loss(
            feat_ref_large, feat_query_small, is_copy, stream, direction
        )

        # Loss when ref < query (incorrect direction)
        loss_incorrect, info_incorrect = asl_loss(
            feat_ref_small, feat_query_large, is_copy, stream, direction
        )

        assert loss_correct < loss_incorrect, (
            f"Loss with correct norm ratio ({loss_correct:.4f}) "
            f"should be less than incorrect ({loss_incorrect:.4f})"
        )

        # Check ratio in info
        assert info_correct["avg_ratio"] > 1.0, "Ratio should be > 1 when ref > query"
        assert info_incorrect["avg_ratio"] < 1.0, "Ratio should be < 1 when ref < query"

    def test_direction_swap_changes_ratio_enforcement(self, asl_loss, device):
        """Verify that swapping direction changes which ratio is enforced."""
        feat_a = torch.randn(4, 768, device=device)
        feat_a = F.normalize(feat_a, p=2, dim=-1) * 2.0  # Norm = 2

        feat_b = torch.randn(4, 768, device=device)
        feat_b = F.normalize(feat_b, p=2, dim=-1) * 1.0  # Norm = 1

        is_copy = torch.ones(4, dtype=torch.bool, device=device)
        stream = ["ndec"] * 4

        # Direction a->b: a is reference
        direction_ab = ["a->b"] * 4
        loss_ab, info_ab = asl_loss(feat_a, feat_b, is_copy, stream, direction_ab)

        # Direction b->a: b is reference (but we pass same features)
        # In practice, the caller would swap feat_a and feat_b based on direction
        direction_ba = ["b->a"] * 4
        loss_ba, info_ba = asl_loss(feat_b, feat_a, is_copy, stream, direction_ba)

        # The losses should be different because norms are swapped
        assert not torch.isclose(
            torch.tensor(loss_ab.item()), torch.tensor(loss_ba.item()), atol=0.01
        ), "Swapping direction should change loss"

    def test_hard_negatives_increase_loss(self, asl_loss, device):
        """Verify that hard negatives (similar but negative) increase loss."""
        # Create similar features
        feat_ref = torch.randn(4, 768, device=device)
        feat_ref = F.normalize(feat_ref, p=2, dim=-1)

        # Query features similar to ref (small perturbation)
        feat_query_similar = feat_ref + torch.randn_like(feat_ref) * 0.1
        feat_query_similar = F.normalize(feat_query_similar, p=2, dim=-1)

        # Query features very different from ref
        feat_query_different = torch.randn(4, 768, device=device)
        feat_query_different = F.normalize(feat_query_different, p=2, dim=-1)

        # Negative pairs
        is_copy = torch.zeros(4, dtype=torch.bool, device=device)
        stream = ["ndec"] * 4
        direction = ["a->b"] * 4

        # Loss for hard negatives (similar features, negative label)
        loss_hard, _ = asl_loss(
            feat_ref, feat_query_similar, is_copy, stream, direction
        )

        # Loss for easy negatives (different features, negative label)
        loss_easy, _ = asl_loss(
            feat_ref, feat_query_different, is_copy, stream, direction
        )

        # Hard negatives should have higher loss (metric loss pushes apart)
        assert loss_hard > loss_easy, (
            f"Hard negative loss ({loss_hard:.4f}) should be greater "
            f"than easy negative loss ({loss_easy:.4f})"
        )

    def test_deterministic_output(self, asl_loss, device):
        """Verify loss computation is deterministic."""
        feat_ref = torch.randn(4, 768, device=device)
        feat_query = torch.randn(4, 768, device=device)
        is_copy = torch.tensor([True, True, False, False], device=device)
        stream = ["disc", "disc", "ndec", "ndec"]
        direction = ["full->crop", "full->crop", "a->b", "a->b"]

        loss_1, info_1 = asl_loss(feat_ref, feat_query, is_copy, stream, direction)
        loss_2, info_2 = asl_loss(feat_ref, feat_query, is_copy, stream, direction)

        assert torch.isclose(loss_1, loss_2), "Loss should be deterministic"
        assert info_1["avg_ratio"] == info_2["avg_ratio"], "Info should be deterministic"

    def test_mixed_stream_handling(self, asl_loss, device):
        """Test handling of mixed DISC and NDEC samples in same batch."""
        batch_size = 8
        feat_ref = torch.randn(batch_size, 768, device=device)
        feat_query = torch.randn(batch_size, 768, device=device)

        # Mix of DISC and NDEC
        is_copy = torch.tensor([True, True, True, True, True, False, False, False], device=device)
        stream = ["disc", "disc", "disc", "disc", "ndec", "ndec", "ndec", "ndec"]
        direction = ["full->crop"] * 4 + ["a->b"] * 4

        loss, info = asl_loss(feat_ref, feat_query, is_copy, stream, direction)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        assert info["disc_count"] == 4
        assert info["ndec_count"] == 4
        assert info["pos_count"] == 5
        assert info["neg_count"] == 3

    def test_similar_pair_for_metric_flag(self, asl_loss, device):
        """Test that similar_pair_for_metric flag affects metric loss behavior."""
        feat_ref = torch.randn(4, 768, device=device)
        feat_ref = F.normalize(feat_ref, p=2, dim=-1)

        # Similar features
        feat_query = feat_ref + torch.randn_like(feat_ref) * 0.1
        feat_query = F.normalize(feat_query, p=2, dim=-1)

        # Negative pairs
        is_copy = torch.zeros(4, dtype=torch.bool, device=device)
        stream = ["ndec"] * 4
        direction = ["a->b"] * 4

        # Without similar_for_metric (default)
        loss_without, _ = asl_loss(
            feat_ref, feat_query, is_copy, stream, direction,
            similar_pair_for_metric=[False, False, False, False]
        )

        # With similar_for_metric (treats as positive for metric)
        loss_with, _ = asl_loss(
            feat_ref, feat_query, is_copy, stream, direction,
            similar_pair_for_metric=[True, True, True, True]
        )

        # When similar_for_metric=True on negative, metric loss treats as positive
        # So similar features should have lower loss
        assert loss_with < loss_without, (
            "similar_pair_for_metric=True should lower metric loss for similar features"
        )


class TestNormRatioLoss:
    """Tests for the NormRatioLoss component."""

    def test_ratio_computation(self, device):
        """Test that norm ratio is computed correctly."""
        from adaptive_dino_icd.losses.metric_losses import NormRatioLoss

        loss_fn = NormRatioLoss(reduction="none")

        # Features with known norms
        feat_ref = torch.zeros(2, 768, device=device)
        feat_ref[0, 0] = 2.0  # Norm = 2
        feat_ref[1, 0] = 1.0  # Norm = 1

        feat_query = torch.zeros(2, 768, device=device)
        feat_query[0, 0] = 1.0  # Norm = 1
        feat_query[1, 0] = 2.0  # Norm = 2

        loss, info = loss_fn(feat_ref, feat_query)

        # First sample: ratio = 2/1 = 2, loss = exp(1-2) = exp(-1) ≈ 0.368
        # Second sample: ratio = 1/2 = 0.5, loss = exp(1-0.5) = exp(0.5) ≈ 1.649
        assert info["avg_ratio"] == pytest.approx(1.25, rel=0.01)  # (2 + 0.5) / 2

    def test_mask_application(self, device):
        """Test that mask correctly filters samples."""
        from adaptive_dino_icd.losses.metric_losses import NormRatioLoss

        loss_fn = NormRatioLoss(reduction="mean")

        feat_ref = torch.randn(4, 768, device=device)
        feat_query = torch.randn(4, 768, device=device)

        # Only apply to first 2 samples
        mask = torch.tensor([True, True, False, False], device=device)

        loss_masked, _ = loss_fn(feat_ref, feat_query, apply_mask=mask)

        # Masked loss should only consider first 2 samples
        assert not torch.isnan(loss_masked)
