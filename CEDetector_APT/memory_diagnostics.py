"""
Memory Diagnostics Script for CEDetector APT Training
Helps identify memory bottlenecks and provides optimization recommendations
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import CEDetectorAPT


def print_gpu_memory(prefix=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"{prefix}GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    else:
        print("No CUDA devices available")


def estimate_model_memory(model):
    """Estimate memory usage of model parameters and buffers."""
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**3
    print(f"Model Parameters: {param_mem:.2f}GB")
    print(f"Model Buffers: {buffer_mem:.2f}GB")
    print(f"Total Model Memory: {param_mem + buffer_mem:.2f}GB")
    return param_mem + buffer_mem


def estimate_activation_memory(batch_size, sequence_length, embed_dim, num_layers):
    """
    Estimate memory for activations during forward/backward pass.

    Without gradient checkpointing, all intermediate activations are stored.
    With gradient checkpointing, only checkpointed layers store activations.
    """
    # Attention activations (Q, K, V projections + attention scores + output)
    attention_per_layer = batch_size * sequence_length * embed_dim * 4  # Q, K, V, output
    attention_scores = batch_size * sequence_length * sequence_length  # attention matrix

    # MLP activations (2 linear layers with 4x expansion)
    mlp_per_layer = batch_size * sequence_length * embed_dim * 4 * 2

    # Total per layer
    per_layer = (attention_per_layer + attention_scores + mlp_per_layer) * 4  # 4 bytes per float32
    total_activations = per_layer * num_layers / 1024**3

    print(f"\nEstimated Activation Memory (FP32):")
    print(f"  Per layer: {per_layer / 1024**3:.3f}GB")
    print(f"  Total ({num_layers} layers): {total_activations:.2f}GB")

    # With gradient checkpointing, reduce by ~50-70%
    with_checkpointing = total_activations * 0.4
    print(f"  With gradient checkpointing: {with_checkpointing:.2f}GB")

    # With FP16, reduce by 50%
    with_fp16 = total_activations * 0.5
    with_both = with_checkpointing * 0.5
    print(f"  With FP16 only: {with_fp16:.2f}GB")
    print(f"  With both (FP16 + checkpointing): {with_both:.2f}GB")

    return total_activations


def test_forward_pass(device_id=0, batch_size=16):
    """Test a forward pass and measure memory usage."""
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Testing Forward Pass (Batch Size: {batch_size})")
    print(f"{'='*60}")

    print_gpu_memory("Before model creation: ")

    # Create model
    model = CEDetectorAPT(
        dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_token_path="/home/jowatson/Deep Learning/Code/.env",
        freeze_backbone=False,
        num_query_patches=6,
        k_neighbors=10
    ).to(device)

    print_gpu_memory("After model creation: ")

    model_mem = estimate_model_memory(model)

    # Create dummy inputs
    query = torch.randn(batch_size, 3, 224, 224, device=device)
    reference = torch.randn(batch_size, 3, 224, 224, device=device)

    print_gpu_memory("After input creation: ")

    try:
        # Forward pass
        model.train()
        outputs = model(query, reference)

        print_gpu_memory("After forward pass: ")

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\nPeak Memory: {peak_mem:.2f}GB")

        # Estimate activations
        # Query: 6 patches per image, each with ~256 tokens
        seq_len = 256 * 6
        estimate_activation_memory(batch_size, seq_len, 768, 4)

        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n{'!'*60}")
            print("OUT OF MEMORY ERROR")
            print(f"{'!'*60}")
            print_gpu_memory("At failure: ")
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"Peak Memory before OOM: {peak_mem:.2f}GB")
            return False
        else:
            raise


def print_optimization_recommendations(oom_batch_size=None):
    """Print recommendations for memory optimization."""
    print(f"\n{'='*60}")
    print("MEMORY OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*60}")

    print("\n1. GRADIENT CHECKPOINTING (Already Implemented)")
    print("   - Saves 50-70% activation memory")
    print("   - Enabled in CopyEditClassifier")
    print("   - Trade-off: ~30% slower training")

    print("\n2. MIXED PRECISION TRAINING (Already Implemented)")
    print("   - FP16 reduces memory by ~50%")
    print("   - Enabled via --use_amp flag")
    print("   - A30 GPUs support TF32 for additional speedup")

    print("\n3. LAYER NORMALIZATION OPTIMIZATION (Already Implemented)")
    print("   - Reuse normalized tensors instead of computing 3x")
    print("   - Saves memory in self-attention layers")

    print("\n4. BATCH SIZE REDUCTION")
    if oom_batch_size:
        print(f"   - Current batch size {oom_batch_size} causes OOM")
        print(f"   - Recommended: {max(oom_batch_size // 2, 4)}")
    else:
        print("   - Current: 12 per GPU (36 total with 3 GPUs)")
        print("   - Can reduce to 8 per GPU if still experiencing OOM")

    print("\n5. PYTORCH MEMORY ALLOCATOR SETTINGS")
    print("   - expandable_segments:True (Already Set)")
    print("   - Reduces fragmentation")
    print("   - Can add garbage_collection_threshold:0.6 for aggressive cleanup")

    print("\n6. BACKBONE FREEZING (Optional)")
    print("   - Freeze DiNOv3 backbone to save ~30% memory")
    print("   - Add --freeze_backbone flag")
    print("   - Trade-off: May reduce final accuracy")

    print("\n7. SEQUENCE LENGTH REDUCTION (Advanced)")
    print("   - Reduce num_query_patches from 6 to 4")
    print("   - Reduces sequence length by 33%")
    print("   - Trade-off: May reduce accuracy on complex images")


def main():
    print("="*60)
    print("CEDetector APT - Memory Diagnostics")
    print("="*60)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA devices available")
        return

    print(f"\nCUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f}GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

    # Test with different batch sizes
    batch_sizes = [16, 12, 8]
    successful_batch = None

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        if test_forward_pass(device_id=0, batch_size=bs):
            successful_batch = bs
            break

    if successful_batch:
        print(f"\n{'='*60}")
        print(f"SUCCESS: Batch size {successful_batch} fits in memory")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("FAILURE: All tested batch sizes cause OOM")
        print(f"{'='*60}")

    print_optimization_recommendations(oom_batch_size=batch_sizes[0] if not successful_batch else None)

    print(f"\n{'='*60}")
    print("Current Configuration:")
    print(f"{'='*60}")
    print("- Gradient Checkpointing: ENABLED")
    print("- Mixed Precision (FP16): ENABLED")
    print("- Layer Norm Optimization: ENABLED")
    print("- Batch Size: 12 per GPU")
    print("- Memory Allocator: expandable_segments=True")
    print("\nThese optimizations should prevent OOM errors on 3x A30 GPUs (24GB each)")


if __name__ == '__main__':
    main()
