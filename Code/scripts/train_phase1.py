#!/usr/bin/env python3
"""
Phase 1 Training Script for Adaptive-DINO-ICD.

Trains the model with:
- DINOv3 backbone (or offline stub for testing)
- APT adaptive tokenization
- ASL loss on mixed DISC/NDEC batches (70/30 ratio)

Run with:
    python scripts/train_phase1.py --config configs/phase1_default.yaml
    python scripts/train_phase1.py --config configs/phase1_offline_stub.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adaptive-DINO-ICD Phase 1")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    return parser.parse_args()


def setup_device(device_arg: Optional[str] = None) -> torch.device:
    """Setup compute device."""
    if device_arg:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def create_dataloaders(config, transform):
    """Create DISC and NDEC dataloaders."""
    from adaptive_dino_icd.data import DISCDataset, NDECDataset, BatchMixer

    # Create datasets
    logger.info(f"Loading DISC dataset from: {config.data.disc_root}")
    disc_dataset = DISCDataset(
        root_dir=config.data.disc_root,
        transform=transform,
        crop_ratio_range=config.data.crop_ratio_range,
    )

    logger.info(f"Loading NDEC dataset from: {config.data.ndec_annotation}")
    ndec_dataset = NDECDataset(
        annotation_file=config.data.ndec_annotation,
        image_root=config.data.ndec_image_root,
        transform=transform,
    )

    # Create batch mixer
    mixer = BatchMixer(
        disc_dataset=disc_dataset,
        ndec_dataset=ndec_dataset,
        batch_size=config.data.batch_size,
        disc_ratio=config.data.disc_ratio,
        num_workers=config.data.num_workers,
        seed=config.training.seed,
    )

    logger.info(f"DISC dataset: {len(disc_dataset)} images")
    logger.info(f"NDEC dataset: {len(ndec_dataset)} pairs")
    logger.info(f"Batch mixer: {len(mixer)} batches per epoch")
    logger.info(f"Batch ratio: {mixer.get_ratio_stats()}")

    return mixer


def train_epoch(
    model: nn.Module,
    loss_module: nn.Module,
    mixer,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    config,
    metrics_logger,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_norm_loss = 0.0
    total_mtr_loss = 0.0
    num_batches = 0

    progress = tqdm(mixer, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(progress):
        # Move to device
        img_ref = batch["img_ref"].to(device)
        img_query = batch["img_query"].to(device)
        is_copy = batch["is_copy"].to(device)
        stream = batch["stream"]
        direction = batch["direction"]
        similar_for_metric = batch["similar_pair_for_metric"]

        # Forward pass through model
        optimizer.zero_grad()

        # Get features for ref and query
        with torch.amp.autocast(device_type=device.type, enabled=False):
            outputs_ref = model(img_ref)
            outputs_query = model(img_query)

            # Compute loss
            loss, info = loss_module(
                outputs_ref["global_feats"],
                outputs_query["global_feats"],
                is_copy,
                stream,
                direction,
                similar_for_metric,
            )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_norm_loss += info["norm_ratio_loss"]
        total_mtr_loss += info["metric_loss"]
        num_batches += 1

        # Update progress bar
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "norm": f"{info['norm_ratio_loss']:.4f}",
            "mtr": f"{info['metric_loss']:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # Log periodically
        if (batch_idx + 1) % config.training.log_interval == 0:
            metrics_logger.log({
                "epoch": epoch,
                "batch": batch_idx + 1,
                "loss": loss.item(),
                "norm_ratio_loss": info["norm_ratio_loss"],
                "metric_loss": info["metric_loss"],
                "avg_ratio": info["avg_ratio"],
                "lr": optimizer.param_groups[0]["lr"],
            })

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_norm = total_norm_loss / num_batches
    avg_mtr = total_mtr_loss / num_batches

    return {
        "loss": avg_loss,
        "norm_ratio_loss": avg_norm,
        "metric_loss": avg_mtr,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config,
    checkpoint_dir: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config.to_dict(),
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def main():
    args = parse_args()

    # Load config
    from adaptive_dino_icd.utils.config import load_config
    config = load_config(args.config)

    # Override seed if specified
    if args.seed is not None:
        config.training.seed = args.seed

    # Set seed
    from adaptive_dino_icd.utils.seed import set_seed
    set_seed(config.training.seed)

    # Setup device
    device = setup_device(args.device)

    # Setup logging
    from adaptive_dino_icd.utils.logging import MetricsLogger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.training.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = MetricsLogger(log_dir / "training_metrics.jsonl")
    logger.info(f"Logging to: {log_dir}")

    # Setup checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints: {checkpoint_dir}")

    # Create transforms
    from adaptive_dino_icd.data import get_train_transforms
    transform = get_train_transforms(
        image_size=config.data.image_size,
        hard_aug_prob=config.data.hard_aug_prob,
        use_augly=config.data.use_augly,
    )

    # Create dataloaders
    mixer = create_dataloaders(config, transform)

    # Create model
    logger.info("Creating model...")
    from adaptive_dino_icd.backbone import AdaptiveBackbone
    from adaptive_dino_icd.utils.config import BackboneConfig, APTConfig

    model = AdaptiveBackbone(
        backbone_config=config.backbone,
        apt_config=config.apt,
        use_adaptive_patches=True,
    )
    model = model.to(device)

    logger.info(f"Model parameters: {model.get_num_params():,}")
    logger.info(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    # Create loss module
    from adaptive_dino_icd.losses import ASLLossModule

    loss_module = ASLLossModule(
        lambda_mtr=config.loss.lambda_mtr,
        temperature=config.loss.temperature,
        margin=config.loss.margin,
        normalize_features=config.loss.normalize_features,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    total_steps = len(mixer) * config.training.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.training.lr * 0.01,
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    # Save config
    from adaptive_dino_icd.utils.config import save_config
    save_config(config, checkpoint_dir / "config.yaml")

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, config.training.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.training.epochs}")

        # Train epoch
        metrics = train_epoch(
            model=model,
            loss_module=loss_module,
            mixer=mixer,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            config=config,
            metrics_logger=metrics_logger,
        )

        # Log epoch metrics
        logger.info(
            f"Epoch {epoch} complete - "
            f"Loss: {metrics['loss']:.4f}, "
            f"Norm: {metrics['norm_ratio_loss']:.4f}, "
            f"Mtr: {metrics['metric_loss']:.4f}"
        )

        metrics_logger.log({
            "epoch": epoch,
            "epoch_loss": metrics["loss"],
            "epoch_norm_loss": metrics["norm_ratio_loss"],
            "epoch_mtr_loss": metrics["metric_loss"],
        })

        # Save checkpoint
        if epoch % config.training.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics, config, checkpoint_dir
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, config.training.epochs, metrics, config, checkpoint_dir
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Final checkpoint: {checkpoint_dir / 'checkpoint_latest.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
