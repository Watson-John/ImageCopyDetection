#!/usr/bin/env python3
"""
Monitor CEDetector Training Progress
Run this script to check the current training status
"""

import json
from pathlib import Path
import sys

def monitor_training(log_file_path, metrics_file_path=None):
    """Monitor training from log file and metrics file."""

    log_file = Path(log_file_path)

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return

    # Read last 30 lines of log
    with open(log_file, 'r') as f:
        lines = f.readlines()

    print("="*70)
    print("TRAINING STATUS")
    print("="*70)

    # Find latest epoch metrics
    epoch_metrics = []
    for line in reversed(lines):
        if "Epoch" in line and "Train Metrics:" in line:
            # Found an epoch summary, get the next few lines
            idx = lines.index(line)
            metrics_text = ''.join(lines[idx:idx+6])
            epoch_metrics.append(metrics_text)
            if len(epoch_metrics) >= 3:  # Get last 3 epochs
                break

    if epoch_metrics:
        print("\n📊 Recent Epoch Results (most recent first):")
        print("-" * 70)
        for i, metrics in enumerate(epoch_metrics):
            print(metrics)
            if i < len(epoch_metrics) - 1:
                print("-" * 70)

    # Get current training line (last non-empty line)
    current_line = None
    for line in reversed(lines):
        if line.strip():
            current_line = line.strip()
            break

    if current_line:
        print("\n⚡ Current Status:")
        print(f"  {current_line}")

    # Read metrics file if available
    if metrics_file_path:
        metrics_file = Path(metrics_file_path)
        if metrics_file.exists():
            print("\n📈 Training Metrics from JSON:")
            print("-" * 70)
            with open(metrics_file, 'r') as f:
                all_metrics = [json.loads(line) for line in f]

            # Print last 5 epochs
            for entry in all_metrics[-5:]:
                epoch = entry.get('epoch', '?')
                phase = entry.get('phase', '?')
                accuracy = entry.get('accuracy', 0)
                loss = entry.get('loss', 0)
                time_min = entry.get('epoch_time_minutes', 0)

                print(f"Epoch {epoch} ({phase}): "
                      f"Acc={accuracy:.4f}, Loss={loss:.4f}, "
                      f"Time={time_min:.1f}min")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
Metric Meanings:
  • µAP (Micro Average Precision):
    - Area under precision-recall curve
    - Target: > 0.85 by end of training
    - Current interpretation:
      * 0.50: Random guessing (very early training)
      * 0.60-0.70: Model is learning basic patterns
      * 0.75-0.85: Good performance, model is working
      * 0.87+: Excellent performance

  • R@P90 (Recall at Precision 90%):
    - Recall when model is 90% confident
    - Target: > 0.80 by end of training
    - Current interpretation:
      * 0.00: Cannot achieve 90% precision (early training)
      * 0.10-0.30: Starting to learn discrimination
      * 0.50-0.70: Good discrimination ability
      * 0.80+: Excellent discrimination

  • Accuracy:
    - Binary classification accuracy at threshold=0.5
    - Target: > 0.90 by end of training
    - Current interpretation:
      * 0.50: Random guessing
      * 0.70-0.80: Learning patterns
      * 0.85-0.92: Good performance

  • Loss:
    - Combined CED loss (contrastive + metric + BCE)
    - Should decrease steadily
    - Early epochs: 5-10 (normal)
    - Mid training: 1-3
    - Late training: < 1

Training Health Indicators:
  ✓ HEALTHY: Loss decreasing, µAP increasing, R@P90 > 0 after epoch 5
  ⚠️  CONCERNING: Loss stuck, µAP not improving, R@P90 = 0 after epoch 5
  ❌ PROBLEM: Loss = NaN, Loss increasing, Accuracy stuck at 0.50

Next Steps:
  • Let training continue for at least 5 epochs
  • Check this monitor every hour or after each epoch
  • R@P90 should become > 0 by epoch 5
  • If R@P90 still 0 after epoch 5, investigate model/data
""")


if __name__ == "__main__":
    # Default paths
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "/home/jowatson/Deep Learning/CEDetector_APT/logs/ced_baseline_gpu4_121522.out"

    if len(sys.argv) > 2:
        metrics_file = sys.argv[2]
    else:
        # Try to infer metrics file from log file
        checkpoint_dir = Path("/home/jowatson/Deep Learning/CEDetector_APT/checkpoints_baseline_20251213_071025")
        metrics_file = checkpoint_dir / "training_metrics.jsonl" if checkpoint_dir.exists() else None

    monitor_training(log_file, metrics_file)

    print("\n💡 TIP: Run this script periodically to monitor training:")
    print(f"   python3 {__file__}")
