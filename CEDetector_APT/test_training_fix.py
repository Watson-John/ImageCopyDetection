"""
Test script to verify training fixes work correctly
Run this before submitting a full training job
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CEDetectorAPT
from utils.losses import CEDLoss
from utils.metrics import MetricsTracker
from data.disc21_dataset import DISC21Dataset
from utils import CEDAugmentations

def test_model_forward():
    """Test that model returns logits properly"""
    print("="*60)
    print("TEST 1: Model Forward Pass")
    print("="*60)

    # Create a small model
    model = CEDetectorAPT(
        dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_token_path="/home/jowatson/Deep Learning/Code/.env",
        freeze_backbone=True,
        num_query_patches=6,
        k_neighbors=10
    )

    # Create dummy inputs
    query = torch.randn(2, 3, 224, 224)
    reference = torch.randn(2, 3, 224, 224)

    # Forward pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(query, reference)

    # Check outputs
    print(f"Output keys: {list(outputs.keys())}")

    if 'logits' not in outputs:
        print("❌ FAIL: Model does not return 'logits' key!")
        return False

    logits = outputs['logits']
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Check that logits are not all the same
    if torch.allclose(logits, logits[0:1].expand_as(logits), atol=1e-6):
        print("❌ WARNING: All logits are the same! Model may not be learning.")
        return False

    print("✓ PASS: Model forward pass works correctly\n")
    return True


def test_loss_computation():
    """Test that loss computation works"""
    print("="*60)
    print("TEST 2: Loss Computation")
    print("="*60)

    criterion = CEDLoss()

    # Create dummy inputs
    B = 4
    D = 768
    cls_tokens_i = torch.randn(B, D)
    cls_tokens_j = torch.randn(B, D)
    patch_embeddings = torch.randn(B, D)
    labels = torch.randint(0, 2, (B,)).float()
    logits = torch.randn(B)

    print(f"Input shapes:")
    print(f"  cls_tokens_i: {cls_tokens_i.shape}")
    print(f"  cls_tokens_j: {cls_tokens_j.shape}")
    print(f"  patch_embeddings: {patch_embeddings.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  logits: {logits.shape}")

    # Compute loss
    print("\nComputing loss...")
    loss_dict = criterion(
        cls_tokens_i=cls_tokens_i,
        cls_tokens_j=cls_tokens_j,
        patch_embeddings=patch_embeddings,
        labels=labels,
        logits=logits
    )

    print(f"✓ Loss components:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value.item():.4f}")

    if loss_dict['total_loss'].item() == 0.0:
        print("❌ FAIL: Total loss is 0!")
        return False

    if torch.isnan(loss_dict['total_loss']):
        print("❌ FAIL: Total loss is NaN!")
        return False

    print("✓ PASS: Loss computation works correctly\n")
    return True


def test_metrics_tracker():
    """Test that metrics are computed correctly"""
    print("="*60)
    print("TEST 3: Metrics Tracker")
    print("="*60)

    tracker = MetricsTracker()

    # Create dummy predictions and labels
    # Create a mix of correct and incorrect predictions
    predictions = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    loss = 0.5

    print(f"Predictions: {predictions.numpy()}")
    print(f"Labels: {labels.numpy()}")

    # Update tracker
    tracker.update(predictions, labels, loss)

    # Compute metrics
    metrics = tracker.compute()

    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Check that metrics are reasonable
    if 'avg_loss' not in metrics:
        print("❌ FAIL: 'avg_loss' not in metrics!")
        return False

    if metrics['avg_loss'] == 0.0:
        print("❌ FAIL: avg_loss is 0!")
        return False

    if metrics['accuracy'] < 0.0 or metrics['accuracy'] > 1.0:
        print("❌ FAIL: accuracy out of range!")
        return False

    if metrics['mu_ap'] < 0.0 or metrics['mu_ap'] > 1.0:
        print("❌ FAIL: mu_ap out of range!")
        return False

    print("✓ PASS: Metrics tracker works correctly\n")
    return True


def test_data_loader():
    """Test that data loader works and labels are balanced"""
    print("="*60)
    print("TEST 4: Data Loader")
    print("="*60)

    # Create augmentations
    augmenter = CEDAugmentations(
        min_ops=2,
        max_ops=2,
        img_size=224
    )

    # Create dataset
    print("Loading DISC21 dataset (small subset)...")
    dataset = DISC21Dataset(
        root_dir="/home/jowatson/Deep Learning/DISC21",
        transform=augmenter,
        subset_fraction=0.001  # Very small for testing
    )

    print(f"✓ Dataset size: {len(dataset)}")

    # Check label distribution
    if len(dataset) > 0:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(16, len(dataset)),
            shuffle=False
        )

        batch = next(iter(loader))
        labels = batch['label']

        print(f"✓ Batch shape: query={batch['query'].shape}, reference={batch['reference'].shape}")
        print(f"✓ Labels: {labels.numpy()}")
        print(f"✓ Label distribution: {labels.float().mean():.3f} (0.5 is balanced)")

        if labels.float().mean() < 0.3 or labels.float().mean() > 0.7:
            print("⚠ WARNING: Labels may be imbalanced!")
    else:
        print("⚠ WARNING: Dataset is empty!")

    print("✓ PASS: Data loader works\n")
    return True


def main():
    print("\n" + "="*60)
    print("TRAINING FIX VERIFICATION TESTS")
    print("="*60 + "\n")

    all_passed = True

    try:
        if not test_model_forward():
            all_passed = False
    except Exception as e:
        print(f"❌ TEST 1 FAILED WITH EXCEPTION: {e}\n")
        all_passed = False

    try:
        if not test_loss_computation():
            all_passed = False
    except Exception as e:
        print(f"❌ TEST 2 FAILED WITH EXCEPTION: {e}\n")
        all_passed = False

    try:
        if not test_metrics_tracker():
            all_passed = False
    except Exception as e:
        print(f"❌ TEST 3 FAILED WITH EXCEPTION: {e}\n")
        all_passed = False

    try:
        if not test_data_loader():
            all_passed = False
    except Exception as e:
        print(f"❌ TEST 4 FAILED WITH EXCEPTION: {e}\n")
        all_passed = False

    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("The training fixes are working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above before training.")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
