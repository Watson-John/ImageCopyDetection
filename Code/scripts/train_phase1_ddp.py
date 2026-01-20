#!/usr/bin/env python3
"""
Phase 1 DDP Training Script for Adaptive-DINO-ICD.

Distributed Data Parallel training across multiple GPUs.

Run with torchrun:
    torchrun --nproc_per_node=3 scripts/train_phase1_ddp.py --config configs/phase1_default.yaml

Or with SLURM:
    srun python scripts/train_phase1_ddp.py --config configs/phase1_default.yaml
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adaptive-DINO-ICD Phase 1 with DDP")
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
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    # DDP arguments (set by torchrun/srun)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training environment."""
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        # Set master address and port for SLURM
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_NODELIST", "localhost").split(",")[0]
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0

    # Initialize process group
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def create_dataloaders_ddp(config, transform, world_size: int, rank: int):
    """Create distributed dataloaders for DISC and NDEC."""
    from adaptive_dino_icd.data import DISCDataset, NDECDataset
    from adaptive_dino_icd.data.batch_mixer import MixedBatchSampler
    from adaptive_dino_icd.data.collate import mixed_collate_fn

    # Create datasets
    disc_dataset = DISCDataset(
        root_dir=config.data.disc_root,
        transform=transform,
        crop_ratio_range=config.data.crop_ratio_range,
    )

    ndec_dataset = NDECDataset(
        annotation_file=config.data.ndec_annotation,
        image_root=config.data.ndec_image_root,
        transform=transform,
    )

    if is_main_process(rank):
        logger.info(f"DISC dataset: {len(disc_dataset)} images")
        logger.info(f"NDEC dataset: {len(ndec_dataset)} pairs")

    # Create distributed samplers
    disc_sampler = DistributedSampler(
        disc_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.training.seed,
    )

    ndec_sampler = DistributedSampler(
        ndec_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.training.seed,
    )

    # Create dataloaders
    # Per-GPU batch size (total effective batch = batch_size * world_size)
    per_gpu_batch = config.data.batch_size
    disc_per_batch = int(per_gpu_batch * config.data.disc_ratio)
    ndec_per_batch = per_gpu_batch - disc_per_batch

    disc_loader = DataLoader(
        disc_dataset,
        batch_size=disc_per_batch,
        sampler=disc_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ndec_loader = DataLoader(
        ndec_dataset,
        batch_size=ndec_per_batch,
        sampler=ndec_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if is_main_process(rank):
        logger.info(f"Per-GPU batch size: {per_gpu_batch} (DISC: {disc_per_batch}, NDEC: {ndec_per_batch})")
        logger.info(f"Effective batch size: {per_gpu_batch * world_size}")

    return disc_loader, ndec_loader, disc_sampler, ndec_sampler


def train_epoch_ddp(
    model: nn.Module,
    loss_module: nn.Module,
    disc_loader: DataLoader,
    ndec_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    config,
    metrics_logger,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Train for one epoch with DDP."""
    model.train()

    total_loss = 0.0
    total_norm_loss = 0.0
    total_mtr_loss = 0.0
    num_batches = 0

    # Create iterators
    disc_iter = iter(disc_loader)
    ndec_iter = iter(ndec_loader)

    # Number of batches per epoch
    num_disc_batches = len(disc_loader)
    num_ndec_batches = len(ndec_loader)
    batches_per_epoch = min(num_disc_batches, num_ndec_batches)

    # Progress bar only on main process
    if is_main_process(rank):
        progress = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}", leave=True)
    else:
        progress = range(batches_per_epoch)

    for batch_idx in progress:
        # Get DISC batch
        try:
            disc_batch = next(disc_iter)
        except StopIteration:
            disc_iter = iter(disc_loader)
            disc_batch = next(disc_iter)

        # Get NDEC batch
        try:
            ndec_batch = next(ndec_iter)
        except StopIteration:
            ndec_iter = iter(ndec_loader)
            ndec_batch = next(ndec_iter)

        # Combine batches
        # DISC: img_full -> img_ref, img_crop -> img_query
        disc_ref = disc_batch["img_full"].to(device)
        disc_query = disc_batch["img_crop"].to(device)
        disc_is_copy = torch.ones(disc_ref.size(0), dtype=torch.bool, device=device)
        disc_stream = ["disc"] * disc_ref.size(0)
        disc_direction = ["full->crop"] * disc_ref.size(0)
        disc_similar = [True] * disc_ref.size(0)

        # NDEC: img_a -> img_ref, img_b -> img_query
        ndec_ref = ndec_batch["img_a"].to(device)
        ndec_query = ndec_batch["img_b"].to(device)
        ndec_is_copy = ndec_batch["is_copy"].to(device)
        ndec_stream = ["ndec"] * ndec_ref.size(0)
        ndec_direction = ndec_batch["direction"]
        ndec_similar = ndec_batch["similar_pair_for_metric"]

        # Concatenate
        img_ref = torch.cat([disc_ref, ndec_ref], dim=0)
        img_query = torch.cat([disc_query, ndec_query], dim=0)
        is_copy = torch.cat([disc_is_copy, ndec_is_copy], dim=0)
        stream = disc_stream + ndec_stream
        direction = disc_direction + list(ndec_direction)
        similar_for_metric = disc_similar + list(ndec_similar)

        # Forward pass
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=True):
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

        # Update progress bar on main process
        if is_main_process(rank):
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "norm": f"{info['norm_ratio_loss']:.4f}",
                "mtr": f"{info['metric_loss']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Log periodically (main process only)
        if is_main_process(rank) and (batch_idx + 1) % config.training.log_interval == 0:
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

    # Reduce metrics across all processes
    if world_size > 1:
        metrics_tensor = torch.tensor([avg_loss, avg_norm, avg_mtr], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss, avg_norm, avg_mtr = metrics_tensor.tolist()

    return {
        "loss": avg_loss,
        "norm_ratio_loss": avg_norm,
        "metric_loss": avg_mtr,
    }


def save_checkpoint_ddp(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config,
    checkpoint_dir: Path,
    rank: int,
):
    """Save training checkpoint (only on main process)."""
    if not is_main_process(rank):
        return

    # Get model state dict (unwrap DDP if necessary)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
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

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Adaptive-DINO-ICD Phase 1 Training (DDP)")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size} GPUs")
        logger.info(f"Device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # Load config
    from adaptive_dino_icd.utils.config import load_config
    config = load_config(args.config)

    # Override seed if specified
    if args.seed is not None:
        config.training.seed = args.seed

    # Set seed (different for each process to ensure different data augmentation)
    from adaptive_dino_icd.utils.seed import set_seed
    set_seed(config.training.seed + rank)

    # Setup logging (main process only)
    if is_main_process(rank):
        from adaptive_dino_icd.utils.logging import MetricsLogger

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(config.training.log_dir) / f"{timestamp}_ddp_{world_size}gpu"
        log_dir.mkdir(parents=True, exist_ok=True)

        metrics_logger = MetricsLogger(log_dir / "training_metrics.jsonl")
        logger.info(f"Logging to: {log_dir}")

        # Setup checkpoint directory
        checkpoint_dir = Path(config.training.checkpoint_dir) / f"{timestamp}_ddp_{world_size}gpu"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints: {checkpoint_dir}")
    else:
        metrics_logger = None
        checkpoint_dir = None

    # Create transforms
    from adaptive_dino_icd.data import get_train_transforms
    transform = get_train_transforms(
        image_size=config.data.image_size,
        hard_aug_prob=config.data.hard_aug_prob,
        use_augly=config.data.use_augly,
    )

    # Create distributed dataloaders
    disc_loader, ndec_loader, disc_sampler, ndec_sampler = create_dataloaders_ddp(
        config, transform, world_size, rank
    )

    # Create model
    if is_main_process(rank):
        logger.info("Creating model...")

    from adaptive_dino_icd.backbone import AdaptiveBackbone

    model = AdaptiveBackbone(
        backbone_config=config.backbone,
        apt_config=config.apt,
        use_adaptive_patches=True,
    )
    model = model.to(device)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Required for adaptive patches
        )

    if is_main_process(rank):
        # Get param count from unwrapped model
        unwrapped = model.module if isinstance(model, DDP) else model
        logger.info(f"Model parameters: {unwrapped.get_num_params():,}")
        logger.info(f"Trainable parameters: {unwrapped.get_num_params(trainable_only=True):,}")

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
    # Estimate total steps
    batches_per_epoch = min(len(disc_loader), len(ndec_loader))
    total_steps = batches_per_epoch * config.training.epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.training.lr * 0.01,
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if is_main_process(rank):
            logger.info(f"Resuming from: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=device)

        # Load model state (handle DDP wrapper)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        if is_main_process(rank):
            logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    # Save config (main process only)
    if is_main_process(rank):
        from adaptive_dino_icd.utils.config import save_config
        save_config(config, checkpoint_dir / "config.yaml")

    # Synchronize before training
    if world_size > 1:
        dist.barrier()

    # Training loop
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Starting DDP training")
        logger.info(f"Epochs: {config.training.epochs}")
        logger.info(f"Batches per epoch: {batches_per_epoch}")
        logger.info(f"Effective batch size: {config.data.batch_size * world_size}")
        logger.info("=" * 60)

    for epoch in range(start_epoch, config.training.epochs + 1):
        # Set epoch for distributed samplers (ensures different shuffling each epoch)
        disc_sampler.set_epoch(epoch)
        ndec_sampler.set_epoch(epoch)

        if is_main_process(rank):
            logger.info(f"\nEpoch {epoch}/{config.training.epochs}")

        # Train epoch
        metrics = train_epoch_ddp(
            model=model,
            loss_module=loss_module,
            disc_loader=disc_loader,
            ndec_loader=ndec_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            config=config,
            metrics_logger=metrics_logger,
            rank=rank,
            world_size=world_size,
        )

        # Log epoch metrics (main process only)
        if is_main_process(rank):
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

        # Save checkpoint (main process only)
        if epoch % config.training.save_interval == 0:
            save_checkpoint_ddp(
                model, optimizer, scheduler, epoch, metrics, config, checkpoint_dir, rank
            )

        # Synchronize after each epoch
        if world_size > 1:
            dist.barrier()

    # Save final checkpoint
    save_checkpoint_ddp(
        model, optimizer, scheduler, config.training.epochs, metrics, config, checkpoint_dir, rank
    )

    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("DDP Training complete!")
        logger.info(f"Final checkpoint: {checkpoint_dir / 'checkpoint_latest.pt'}")
        logger.info("=" * 60)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
