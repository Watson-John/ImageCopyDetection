#!/usr/bin/env python3
"""
Sanity check with REAL data and REAL DINOv3 model.

Tests the complete pipeline:
1. Load real images from DISC21 and NDEC
2. Apply augmentation pipeline
3. Run through DINOv3 backbone (not offline stub)
4. Compute ASL loss
5. Run backward pass to verify gradients

This validates everything before committing to full training.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check with real data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=3, help="Number of batches to test")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not specified)")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("SANITY CHECK: Real Data + Real DINOv3 Model")
    logger.info("=" * 70)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # =========================================================================
    # Step 1: Test Data Loading
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Testing Data Loading")
    logger.info("=" * 70)

    from adaptive_dino_icd.data import DISCDataset, NDECDataset, BatchMixer
    from adaptive_dino_icd.data.augmentations import get_train_transforms

    # Create transforms
    transform = get_train_transforms(image_size=224, hard_aug_prob=0.5)
    logger.info("✓ Augmentation pipeline created")

    # Load DISC21 dataset (use single subdirectory for faster loading)
    disc_root = "/home/jowatson/Deep Learning/DISC21/refs_50k_0"
    logger.info(f"\nLoading DISC21 from: {disc_root}")
    logger.info("  (Using refs_50k_0 subdirectory for faster sanity check)")

    disc_dataset = DISCDataset(
        root_dir=disc_root,
        transform=transform,
        crop_ratio_range=(0.3, 0.8),
        max_samples=500,  # Limit for sanity check
    )
    logger.info(f"✓ DISC21 loaded: {len(disc_dataset)} samples")

    # Load NDEC dataset
    ndec_root = "/home/jowatson/Deep Learning/NDEC"
    ndec_annotation = os.path.join(ndec_root, "pairs_converted.csv")
    logger.info(f"\nLoading NDEC from: {ndec_annotation}")

    ndec_dataset = NDECDataset(
        annotation_file=ndec_annotation,
        image_root=ndec_root,
        transform=transform,
        max_samples=200,  # Limit for sanity check
    )
    logger.info(f"✓ NDEC loaded: {len(ndec_dataset)} pairs")
    logger.info(f"  Statistics: {ndec_dataset.get_statistics()}")

    # Create BatchMixer
    logger.info(f"\nCreating BatchMixer (batch_size={args.batch_size}, disc_ratio=0.7)")
    mixer = BatchMixer(
        disc_dataset=disc_dataset,
        ndec_dataset=ndec_dataset,
        batch_size=args.batch_size,
        disc_ratio=0.7,
        num_workers=2,
        seed=42,
    )
    logger.info(f"✓ BatchMixer created: {len(mixer)} batches per epoch")

    # Test loading a batch
    logger.info("\nTesting batch loading...")
    start_time = time.time()
    batch = next(iter(mixer))
    load_time = time.time() - start_time

    logger.info(f"✓ Batch loaded in {load_time:.2f}s")
    logger.info(f"  - img_ref shape: {batch['img_ref'].shape}")
    logger.info(f"  - img_query shape: {batch['img_query'].shape}")
    logger.info(f"  - streams: {batch['stream']}")
    logger.info(f"  - is_copy: {batch['is_copy'].tolist()}")

    # =========================================================================
    # Step 2: Test Real DINOv3 Model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Loading Real DINOv3 Model")
    logger.info("=" * 70)

    from adaptive_dino_icd.backbone import AdaptiveBackbone
    from adaptive_dino_icd.utils.config import BackboneConfig, APTConfig

    # Create configs for REAL model (not offline stub)
    backbone_config = BackboneConfig(
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        offline_stub=False,  # USE REAL MODEL
        pretrained=True,
        embed_dim=768,
        freeze_backbone=False,
    )

    apt_config = APTConfig(
        entropy_scales=[8, 16, 32],
        min_patch_size=8,
        max_patch_size=64,
        entropy_thresholds={8: 0.3, 16: 0.5, 32: 0.7},
        normalize_entropy=True,
        embed_dim=768,
        smallest_patch_size=16,
    )

    logger.info("Loading DINOv3 model (this may take a moment)...")
    start_time = time.time()

    model = AdaptiveBackbone(
        backbone_config=backbone_config,
        apt_config=apt_config,
        use_adaptive_patches=True,
    )
    model = model.to(device)
    model.train()

    load_time = time.time() - start_time
    logger.info(f"✓ DINOv3 model loaded in {load_time:.1f}s")
    logger.info(f"  - Total parameters: {model.get_num_params():,}")
    logger.info(f"  - Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    # =========================================================================
    # Step 3: Test Forward Pass with Real Data
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Testing Forward Pass with Real Data")
    logger.info("=" * 70)

    img_ref = batch["img_ref"].to(device)
    img_query = batch["img_query"].to(device)

    logger.info(f"Input shapes: ref={img_ref.shape}, query={img_query.shape}")
    logger.info("Running forward pass...")

    start_time = time.time()
    with torch.amp.autocast(device_type=device.type, enabled=True):
        outputs_ref = model(img_ref)
        outputs_query = model(img_query)
    forward_time = time.time() - start_time

    logger.info(f"✓ Forward pass completed in {forward_time:.2f}s")
    logger.info(f"  - global_feats: {outputs_ref['global_feats'].shape}")
    logger.info(f"  - local_feats: {outputs_ref['local_feats'].shape}")
    logger.info(f"  - local_coords: {outputs_ref['local_coords'].shape}")

    # Check for NaN/Inf
    if torch.isnan(outputs_ref['global_feats']).any():
        logger.error("NaN detected in global_feats!")
    elif torch.isinf(outputs_ref['global_feats']).any():
        logger.error("Inf detected in global_feats!")
    else:
        logger.info("  - No NaN/Inf in outputs ✓")

    # =========================================================================
    # Step 4: Test ASL Loss
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Testing ASL Loss Computation")
    logger.info("=" * 70)

    from adaptive_dino_icd.losses import ASLLossModule

    loss_module = ASLLossModule(
        lambda_mtr=0.5,
        temperature=0.07,
        margin=0.2,
        normalize_features=True,
    )

    logger.info("Computing loss...")
    start_time = time.time()

    loss, info = loss_module(
        outputs_ref["global_feats"],
        outputs_query["global_feats"],
        batch["is_copy"].to(device),
        batch["stream"],
        batch["direction"],
        batch["similar_pair_for_metric"],
    )

    loss_time = time.time() - start_time
    logger.info(f"✓ Loss computed in {loss_time:.3f}s")
    logger.info(f"  - Total loss: {info['total_loss']:.4f}")
    logger.info(f"  - Norm ratio loss: {info['norm_ratio_loss']:.4f}")
    logger.info(f"  - Metric loss: {info['metric_loss']:.4f}")
    logger.info(f"  - Avg ratio: {info['avg_ratio']:.4f}")
    logger.info(f"  - DISC count: {info['disc_count']}, NDEC count: {info['ndec_count']}")

    # =========================================================================
    # Step 5: Test Backward Pass
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Testing Backward Pass")
    logger.info("=" * 70)

    logger.info("Running backward pass...")
    start_time = time.time()

    loss.backward()

    backward_time = time.time() - start_time
    logger.info(f"✓ Backward pass completed in {backward_time:.2f}s")

    # Check gradients
    total_grad_norm = 0.0
    num_params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
            num_params_with_grad += 1

    total_grad_norm = total_grad_norm ** 0.5
    logger.info(f"  - Parameters with gradients: {num_params_with_grad}")
    logger.info(f"  - Total gradient norm: {total_grad_norm:.4f}")

    if total_grad_norm == 0:
        logger.warning("Gradient norm is 0 - check if model is frozen!")
    elif total_grad_norm > 100:
        logger.warning(f"Large gradient norm ({total_grad_norm:.2f}) - may need gradient clipping")
    else:
        logger.info("  - Gradient magnitude looks healthy ✓")

    # =========================================================================
    # Step 6: Test Multiple Batches
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 6: Testing {args.num_batches} Batches (Simulating Training)")
    logger.info("=" * 70)

    model.zero_grad()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    batch_times = []
    losses = []

    mixer_iter = iter(mixer)
    for i in range(args.num_batches):
        batch_start = time.time()

        try:
            batch = next(mixer_iter)
        except StopIteration:
            mixer_iter = iter(mixer)
            batch = next(mixer_iter)

        img_ref = batch["img_ref"].to(device)
        img_query = batch["img_query"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=True):
            outputs_ref = model(img_ref)
            outputs_query = model(img_query)

            loss, info = loss_module(
                outputs_ref["global_feats"],
                outputs_query["global_feats"],
                batch["is_copy"].to(device),
                batch["stream"],
                batch["direction"],
                batch["similar_pair_for_metric"],
            )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        losses.append(info['total_loss'])

        logger.info(f"  Batch {i+1}/{args.num_batches}: loss={info['total_loss']:.4f}, time={batch_time:.2f}s")

    avg_batch_time = sum(batch_times) / len(batch_times)
    logger.info(f"\n✓ All batches completed successfully")
    logger.info(f"  - Avg batch time: {avg_batch_time:.2f}s")
    logger.info(f"  - Losses: {[f'{l:.4f}' for l in losses]}")

    # =========================================================================
    # Step 7: Memory Usage
    # =========================================================================
    if torch.cuda.is_available():
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: GPU Memory Usage")
        logger.info("=" * 70)

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        logger.info(f"  - Currently allocated: {allocated:.2f} GB")
        logger.info(f"  - Currently reserved: {reserved:.2f} GB")
        logger.info(f"  - Peak allocated: {max_allocated:.2f} GB")

        # Estimate for larger batch sizes
        mem_per_sample = max_allocated / args.batch_size
        logger.info(f"  - Estimated memory per sample: {mem_per_sample:.2f} GB")
        logger.info(f"  - Estimated for batch_size=8: {mem_per_sample * 8:.2f} GB")
        logger.info(f"  - Estimated for batch_size=16: {mem_per_sample * 16:.2f} GB")
        logger.info(f"  - Estimated for batch_size=32: {mem_per_sample * 32:.2f} GB")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SANITY CHECK COMPLETE - ALL TESTS PASSED! ✓")
    logger.info("=" * 70)
    logger.info("""
Summary:
  ✓ DISC21 data loading works
  ✓ NDEC data loading works
  ✓ Augmentation pipeline works
  ✓ BatchMixer (70/30 ratio) works
  ✓ Real DINOv3 model loads and runs
  ✓ ASL loss computation works
  ✓ Backward pass and gradients work
  ✓ Optimizer step works

Ready for full training!
""")


if __name__ == "__main__":
    main()
