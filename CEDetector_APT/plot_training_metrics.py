"""
Plot Training Metrics from JSONL Log File
Creates loss and accuracy curves over epochs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_metrics(log_file):
    """Load metrics from JSONL file."""
    metrics = []
    with open(log_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line.strip()))
    return metrics


def plot_metrics(metrics, output_dir='./plots'):
    """Generate plots for loss and accuracy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data
    epochs = [m['epoch'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    accuracies = [m.get('accuracy', 0) for m in metrics]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Plot Accuracy
    ax2.plot(epochs, accuracies, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    # Create separate plots
    # Loss only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    loss_path = Path(output_dir) / 'loss_curve.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss plot to: {loss_path}")
    plt.close()

    # Accuracy only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim([0, 1])
    acc_path = Path(output_dir) / 'accuracy_curve.png'
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to: {acc_path}")
    plt.close()

    # Print statistics
    print(f"\nTraining Statistics:")
    print(f"  Total Epochs: {len(epochs)}")
    print(f"  Initial Loss: {losses[0]:.4f}")
    print(f"  Final Loss: {losses[-1]:.4f}")
    print(f"  Loss Reduction: {losses[0] - losses[-1]:.4f} ({100*(losses[0]-losses[-1])/losses[0]:.1f}%)")
    print(f"  Best Loss: {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})")
    if accuracies:
        print(f"  Initial Accuracy: {accuracies[0]:.4f}")
        print(f"  Final Accuracy: {accuracies[-1]:.4f}")
        print(f"  Best Accuracy: {max(accuracies):.4f} (Epoch {epochs[accuracies.index(max(accuracies))]})")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from JSONL log')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training_metrics.jsonl file')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    args = parser.parse_args()

    # Load and plot metrics
    print(f"Loading metrics from: {args.log_file}")
    metrics = load_metrics(args.log_file)
    print(f"Loaded {len(metrics)} metric entries")

    plot_metrics(metrics, args.output_dir)


if __name__ == '__main__':
    main()
