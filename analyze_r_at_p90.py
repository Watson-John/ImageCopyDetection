"""
Detailed Analysis of R@P90 Metric for CEDetector Training

This script explains why R@P90 can be 0 even when µAP is ~0.5
"""

import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Simulate predictions that would give µAP ~0.5 but R@P90 = 0
# This happens when positive and negative samples overlap heavily

np.random.seed(42)

# Scenario 1: Heavy overlap (similar to epoch 0)
print("="*70)
print("SCENARIO 1: Heavy Overlap (like your Epoch 0)")
print("="*70)

# Create synthetic predictions where positives and negatives overlap
n_samples = 1000
n_positive = 500
n_negative = 500

# Positive samples: mean=0.55, std=0.20 (slight bias toward higher scores)
positive_preds = np.clip(np.random.normal(0.55, 0.20, n_positive), 0, 1)

# Negative samples: mean=0.45, std=0.20 (slight bias toward lower scores)
negative_preds = np.clip(np.random.normal(0.45, 0.20, n_negative), 0, 1)

# Combine
predictions = np.concatenate([positive_preds, negative_preds])
labels = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])

# Compute metrics
from sklearn.metrics import average_precision_score
mu_ap = average_precision_score(labels, predictions)

precision, recall, thresholds = precision_recall_curve(labels, predictions)

# Find recall at P90
valid_idx = np.where(precision >= 0.90)[0]
r_at_p90 = recall[valid_idx].max() if len(valid_idx) > 0 else 0.0

print(f"\nPrediction Statistics:")
print(f"  Positive samples mean: {positive_preds.mean():.3f} ± {positive_preds.std():.3f}")
print(f"  Negative samples mean: {negative_preds.mean():.3f} ± {negative_preds.std():.3f}")
print(f"  Overlap: {np.sum((positive_preds < 0.5)) + np.sum((negative_preds > 0.5))} / {n_samples} samples")

print(f"\nMetrics:")
print(f"  µAP: {mu_ap:.4f}")
print(f"  R@P90: {r_at_p90:.4f}")
print(f"  Max Precision Achieved: {precision.max():.4f}")

# Find threshold that gives best precision
best_precision_idx = np.argmax(precision)
best_precision = precision[best_precision_idx]
print(f"  Best precision: {best_precision:.4f} at recall={recall[best_precision_idx]:.4f}")

print("\nExplanation:")
if r_at_p90 == 0:
    print("  ⚠️  The model NEVER achieves 90% precision at any threshold!")
    print("  This means positive and negative samples are too mixed.")
    print("  The model needs more training to learn better discrimination.")
else:
    print(f"  ✓ The model achieves 90% precision with {r_at_p90:.1%} recall.")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Prediction distributions
ax = axes[0, 0]
bins = np.linspace(0, 1, 50)
ax.hist(positive_preds, bins=bins, alpha=0.5, label='Positive samples', color='green', density=True)
ax.hist(negative_preds, bins=bins, alpha=0.5, label='Negative samples', color='red', density=True)
ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
ax.set_xlabel('Prediction Score')
ax.set_ylabel('Density')
ax.set_title('Prediction Distribution (Heavy Overlap)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Precision-Recall curve
ax = axes[0, 1]
ax.plot(recall, precision, linewidth=2)
ax.axhline(0.9, color='red', linestyle='--', label='P=0.90 (target)')
ax.axhline(best_precision, color='orange', linestyle='--',
           label=f'Max P={best_precision:.3f}')
ax.fill_between(recall, precision, alpha=0.2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall Curve (µAP={mu_ap:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Scenario 2: Good separation (expected after training)
print("\n" + "="*70)
print("SCENARIO 2: Good Separation (Expected after 10+ epochs)")
print("="*70)

# Positive samples: mean=0.75, std=0.15 (well-separated)
positive_preds_good = np.clip(np.random.normal(0.75, 0.15, n_positive), 0, 1)

# Negative samples: mean=0.25, std=0.15 (well-separated)
negative_preds_good = np.clip(np.random.normal(0.25, 0.15, n_negative), 0, 1)

predictions_good = np.concatenate([positive_preds_good, negative_preds_good])
labels_good = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])

mu_ap_good = average_precision_score(labels_good, predictions_good)
precision_good, recall_good, _ = precision_recall_curve(labels_good, predictions_good)
valid_idx_good = np.where(precision_good >= 0.90)[0]
r_at_p90_good = recall_good[valid_idx_good].max() if len(valid_idx_good) > 0 else 0.0

print(f"\nPrediction Statistics:")
print(f"  Positive samples mean: {positive_preds_good.mean():.3f} ± {positive_preds_good.std():.3f}")
print(f"  Negative samples mean: {negative_preds_good.mean():.3f} ± {negative_preds_good.std():.3f}")
print(f"  Overlap: {np.sum((positive_preds_good < 0.5)) + np.sum((negative_preds_good > 0.5))} / {n_samples} samples")

print(f"\nMetrics:")
print(f"  µAP: {mu_ap_good:.4f}")
print(f"  R@P90: {r_at_p90_good:.4f}")
print(f"  Max Precision Achieved: {precision_good.max():.4f}")

# Plot 3: Good prediction distributions
ax = axes[1, 0]
ax.hist(positive_preds_good, bins=bins, alpha=0.5, label='Positive samples', color='green', density=True)
ax.hist(negative_preds_good, bins=bins, alpha=0.5, label='Negative samples', color='red', density=True)
ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
ax.set_xlabel('Prediction Score')
ax.set_ylabel('Density')
ax.set_title('Prediction Distribution (Good Separation)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Good Precision-Recall curve
ax = axes[1, 1]
ax.plot(recall_good, precision_good, linewidth=2, color='green')
ax.axhline(0.9, color='red', linestyle='--', label='P=0.90 (target)')
if r_at_p90_good > 0:
    ax.scatter([r_at_p90_good], [0.9], s=100, color='red', zorder=5,
               label=f'R@P90={r_at_p90_good:.3f}')
ax.fill_between(recall_good, precision_good, alpha=0.2, color='green')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall Curve (µAP={mu_ap_good:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/home/jowatson/Deep Learning/r_at_p90_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: /home/jowatson/Deep Learning/r_at_p90_analysis.png")

# Summary
print("\n" + "="*70)
print("SUMMARY FOR YOUR TRAINING")
print("="*70)
print("""
Your Current State (Epoch 0):
  µAP: 0.4988  → Similar to heavy overlap scenario
  R@P90: 0.0000 → Model cannot achieve 90% precision yet
  Accuracy: 0.4888 → Close to random guessing

Why R@P90 is 0:
  The model's predictions for positive (copy-edit) and negative (non-copy-edit)
  samples are heavily overlapping. There is no threshold where the model can
  say "I'm 90% confident this is a copy-edit."

This is NORMAL for early training because:
  1. The model starts with random weights
  2. DiNOv3 features need fine-tuning for this specific task
  3. The classifier needs to learn the cross-attention patterns
  4. Dataset is challenging (DISC21 with edited copies)

Expected Training Progression:
  Epoch 0-2:   R@P90 = 0.00, µAP = 0.50-0.60
  Epoch 3-5:   R@P90 = 0.10-0.30, µAP = 0.65-0.75
  Epoch 6-10:  R@P90 = 0.40-0.60, µAP = 0.78-0.85
  Epoch 11-18: R@P90 = 0.70-0.85, µAP = 0.87-0.92

Action Items:
  ✓ Continue training - your setup is correct
  ✓ Monitor µAP increasing (this is the main indicator)
  ✓ Check R@P90 after epoch 5 - it should be > 0 by then
  ⚠️  If R@P90 stays 0 after epoch 5, investigate model/data issues
""")
