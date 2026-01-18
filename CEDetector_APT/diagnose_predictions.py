"""
Diagnostic script to analyze model predictions and identify why R@P90 is 0.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Check if checkpoint exists
checkpoint_dirs = list(Path('.').glob('checkpoints_*'))
if not checkpoint_dirs:
    print("No checkpoints directories found!")
    exit(1)

# Get most recent checkpoint directory
latest_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
print(f"Using checkpoint directory: {latest_dir}")

# Find most recent checkpoint
checkpoints = list(latest_dir.glob('checkpoint_epoch_*.pt'))
if not checkpoints:
    print(f"No checkpoints found in {latest_dir}!")
    exit(1)

# Load the most recent checkpoint
latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"Loading checkpoint: {latest_checkpoint}")

checkpoint = torch.load(latest_checkpoint, map_location='cpu')

print("\n" + "="*60)
print("CHECKPOINT ANALYSIS")
print("="*60)

# Check what's in the checkpoint
print("\nCheckpoint contents:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Analyze training metrics if available
if 'train_metrics' in checkpoint:
    metrics = checkpoint['train_metrics']
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

print("\n" + "="*60)
print("MODEL ARCHITECTURE ANALYSIS")
print("="*60)

# Load model state dict
state_dict = checkpoint['model_state_dict']

# Find classifier weights (the final layer that outputs logits)
classifier_keys = [k for k in state_dict.keys() if 'classifier' in k.lower() or 'head' in k.lower()]

print(f"\nClassifier layers found: {len(classifier_keys)}")
for key in classifier_keys:
    tensor = state_dict[key]
    print(f"\n{key}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")

print("\n" + "="*60)
print("SIMULATION: Prediction Distribution Analysis")
print("="*60)

# Simulate what predictions might look like
# Generate random logits similar to what the model might output
np.random.seed(42)

# Scenario 1: Model outputs near-zero logits (not confident)
logits_scenario1 = np.random.randn(1000) * 0.1  # Small variance
probs_scenario1 = 1 / (1 + np.exp(-logits_scenario1))  # Sigmoid

# Scenario 2: Model outputs random logits
logits_scenario2 = np.random.randn(1000) * 1.0  # Normal variance
probs_scenario2 = 1 / (1 + np.exp(-logits_scenario2))

# Scenario 3: Model has learned some signal
logits_scenario3 = np.concatenate([
    np.random.randn(500) * 0.5 - 1.0,  # Negative examples
    np.random.randn(500) * 0.5 + 1.0   # Positive examples
])
probs_scenario3 = 1 / (1 + np.exp(-logits_scenario3))

print("\nScenario 1: Near-zero logits (not confident)")
print(f"  Logits - Mean: {logits_scenario1.mean():.4f}, Std: {logits_scenario1.std():.4f}")
print(f"  Probs  - Mean: {probs_scenario1.mean():.4f}, Std: {probs_scenario1.std():.4f}")
print(f"  Probs  - Min: {probs_scenario1.min():.4f}, Max: {probs_scenario1.max():.4f}")

print("\nScenario 2: Random logits")
print(f"  Logits - Mean: {logits_scenario2.mean():.4f}, Std: {logits_scenario2.std():.4f}")
print(f"  Probs  - Mean: {probs_scenario2.mean():.4f}, Std: {probs_scenario2.std():.4f}")
print(f"  Probs  - Min: {probs_scenario2.min():.4f}, Max: {probs_scenario2.max():.4f}")

print("\nScenario 3: Model has learned signal")
print(f"  Logits - Mean: {logits_scenario3.mean():.4f}, Std: {logits_scenario3.std():.4f}")
print(f"  Probs  - Mean: {probs_scenario3.mean():.4f}, Std: {probs_scenario3.std():.4f}")
print(f"  Probs  - Min: {probs_scenario3.min():.4f}, Max: {probs_scenario3.max():.4f}")

# Calculate R@P90 for each scenario
from sklearn.metrics import precision_recall_curve

def compute_r_at_p90(predictions, labels):
    """Compute R@P90"""
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    valid_idx = np.where(precision >= 0.9)[0]
    if len(valid_idx) == 0:
        return 0.0
    return recall[valid_idx].max()

# Create balanced labels
labels = np.concatenate([np.zeros(500), np.ones(500)])

r_at_p90_s1 = compute_r_at_p90(probs_scenario1, labels)
r_at_p90_s2 = compute_r_at_p90(probs_scenario2, labels)
r_at_p90_s3 = compute_r_at_p90(probs_scenario3, labels)

print("\n" + "="*60)
print("R@P90 for each scenario:")
print(f"  Scenario 1 (not confident): {r_at_p90_s1:.4f}")
print(f"  Scenario 2 (random): {r_at_p90_s2:.4f}")
print(f"  Scenario 3 (learned signal): {r_at_p90_s3:.4f}")
print("="*60)

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

print("\nBased on your metrics:")
print("  µAP: 0.4939 (below 0.5 - worse than random)")
print("  R@P90: 0.0000 (no confident predictions)")
print("  Accuracy: 0.4884 (below 50% - worse than random)")
print("  Loss: 1.1592")

print("\nMost likely issues:")
print("  1. Model outputs are centered around 0.5 probability (no confidence)")
print("  2. Model might not be learning proper signal")
print("  3. Possible issues:")
print("     - Loss function not working correctly")
print("     - Gradient flow issues")
print("     - Label mismatch (positive/negative flipped?)")
print("     - Learning rate too high or too low")
print("     - Architecture not connecting properly")

print("\nRecommended actions:")
print("  1. Check if logits are all near 0 (sigmoid → 0.5)")
print("  2. Verify labels are correct in dataset")
print("  3. Check if loss is decreasing over epochs")
print("  4. Inspect gradient magnitudes")
print("  5. Try simpler baseline model first")
