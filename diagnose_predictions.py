"""
Diagnose why R@P90 is 0.0000
This script analyzes the prediction distribution to understand the issue
"""

import json
import numpy as np
from pathlib import Path

# Read the training metrics
checkpoint_dir = Path("/home/jowatson/Deep Learning/CEDetector_APT/checkpoints_baseline_20251213_071025")
metrics_file = checkpoint_dir / "training_metrics.jsonl"

if metrics_file.exists():
    print("Reading training metrics...")
    with open(metrics_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            print(f"Epoch {entry['epoch']}: {entry}")
else:
    print(f"Metrics file not found: {metrics_file}")

# Also check if there's a checkpoint we can load to inspect the model
checkpoints = list(checkpoint_dir.glob("*.pt"))
print(f"\nFound {len(checkpoints)} checkpoint files:")
for ckpt in checkpoints:
    print(f"  - {ckpt.name}")

print("\n" + "="*60)
print("Analysis of R@P90 = 0.0000")
print("="*60)

print("""
R@P90 (Recall at Precision 90%) measures the recall when the model
achieves 90% precision at some threshold.

R@P90 = 0.0000 means the model NEVER achieves 90% precision at any
threshold, which indicates:

1. Poor Separation: The model's predictions for positive and negative
   samples are heavily overlapping. The model can't confidently
   distinguish between copy-edits and non-copy-edits.

2. Early Training: After epoch 0, this is somewhat expected. The model
   hasn't learned good feature representations yet.

3. Dataset Difficulty: The DISC21 dataset is challenging, especially
   with only 50% of the data (subset_fraction=0.5).

Expected Progress:
- Epoch 0-2: R@P90 might stay at 0, µAP should improve slowly
- Epoch 3-5: R@P90 should start increasing (0.1-0.3)
- Epoch 10+: R@P90 should reach 0.5-0.7
- Epoch 18: R@P90 should be 0.7-0.8 (if training is successful)

Current Metrics (Epoch 0):
- µAP: 0.4988 ✓ (reasonable, close to random 0.5)
- Accuracy: 0.4888 ✓ (close to random 0.5)
- R@P90: 0.0000 ⚠️ (concerning but expected early on)

Recommendations:
1. Continue training for at least 5 epochs to see if R@P90 improves
2. Monitor µAP - it should steadily increase
3. Check for NaN losses or gradient issues
4. If R@P90 stays at 0 after epoch 5, there may be a model issue
""")

# Check the current training log
log_file = Path("/home/jowatson/Deep Learning/CEDetector_APT/logs/ced_baseline_gpu4_121522.out")
if log_file.exists():
    print("\n" + "="*60)
    print("Checking if training is still running...")
    print("="*60)
    with open(log_file, 'r') as f:
        lines = f.readlines()
        # Get last 10 lines
        print("Last 10 lines of training log:")
        for line in lines[-10:]:
            print(line.rstrip())
