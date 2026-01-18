"""
Diagnostic script to identify training issues
"""

import torch
import json
import numpy as np

# Load recent metrics
metrics_file = "./checkpoints_20251213_103418/training_metrics.jsonl"

print("="*60)
print("TRAINING DIAGNOSTICS")
print("="*60)

# Analyze metrics
with open(metrics_file, 'r') as f:
    metrics = [json.loads(line) for line in f]

print(f"\nTotal epochs logged: {len(metrics)}")
print(f"\nFirst 5 epochs:")
for m in metrics[:5]:
    print(f"  Epoch {m['epoch']}: Loss={m['loss']:.4f}, Acc={m['accuracy']:.4f}")

print(f"\nLast 5 epochs:")
for m in metrics[-5:]:
    print(f"  Epoch {m['epoch']}: Loss={m['loss']:.4f}, Acc={m['accuracy']:.4f}")

# Check for loss issues
losses = [m['loss'] for m in metrics]
accuracies = [m['accuracy'] for m in metrics]

print(f"\n{'='*60}")
print("ISSUE DETECTION")
print("="*60)

# Issue 1: All losses are zero
if all(l == 0.0 for l in losses):
    print("\n❌ CRITICAL ISSUE: All losses are 0.0!")
    print("   This means the loss values are not being tracked properly.")

# Issue 2: Accuracy around 50%
early_acc = np.mean(accuracies[:10])
if 0.45 < early_acc < 0.55:
    print(f"\n❌ CRITICAL ISSUE: Early accuracy ~{early_acc:.3f} (random guessing)")
    print("   This suggests predictions are all similar (likely all 0.5)")
    print("   Probable cause: logits are defaulting to zeros")

# Issue 3: Sudden accuracy jump
if len(accuracies) > 15:
    mid_acc = np.mean(accuracies[10:15])
    late_acc = np.mean(accuracies[-5:])
    if late_acc - mid_acc > 0.4:
        print(f"\n❌ SUSPICIOUS: Accuracy jumped from {mid_acc:.3f} to {late_acc:.3f}")
        print("   This suggests a major change in behavior or data")

print(f"\n{'='*60}")
print("ROOT CAUSE ANALYSIS")
print("="*60)

print("""
Based on the diagnostics, the likely issues are:

1. **Loss tracking bug**: The loss values in metrics are 0.0 because:
   - MetricsTracker.compute() returns metrics with 'avg_loss' key
   - But the training code logs 'loss' key which doesn't exist

2. **Logits defaulting to zeros**: In train.py line 93:
   - logits = outputs.get('logits', torch.zeros(...))
   - If the model doesn't return 'logits', it defaults to zeros
   - torch.sigmoid(zeros) = 0.5 for all samples
   - This causes ~50% accuracy and R@P90 = 0

3. **Model forward() not returning logits**: The CEDetectorAPT model
   might not be returning the 'logits' key properly.

SOLUTION:
1. Fix the loss logging key mismatch
2. Ensure model always returns 'logits'
3. Add validation to catch this issue early
4. Check label distribution
""")

print(f"\n{'='*60}")
print("RECOMMENDED FIXES")
print("="*60)
print("""
1. In train.py line 158: Change 'loss' to 'avg_loss'
2. In train.py line 93: Remove the default zeros, raise error instead
3. Add debug prints to verify logits are not all the same
4. Check if classifier is being called properly
""")
