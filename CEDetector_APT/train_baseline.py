"""
Training Script for Baseline CEDetector (without APT)
Supports DDP for multi-GPU training
"""

import os
import argparse
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime, timedelta

# Filter known warnings
warnings.filterwarnings('ignore', message='.*output_attentions.*attn_implementation.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.*')

# Local imports
from models.cedetector_baseline import CEDetectorBaseline
from data.disc21_dataset import DISC21Dataset
from data.ndec_dataset import NDECDataset
from utils import CEDAugmentations
from utils.losses import CEDLoss
from utils.metrics import MetricsTracker


def setup_ddp():
    """Initialize DDP with NCCL backend for NVIDIA GPUs."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)

    # Enable TF32 for A30 (Ampere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        print(f"DDP initialized: {world_size} GPUs (NCCL backend)")
        print(f"Baseline CED (no APT) - TF32 enabled for Ampere GPUs")

    return rank, local_rank, world_size


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def train_epoch(model, loader, criterion, optimizer, epoch, device, rank, scaler=None, log_file=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    tracker = MetricsTracker()
    use_amp = scaler is not None

    # Track timing for ETA
    epoch_start_time = time.time()
    batch_times = []

    for batch_idx, batch in enumerate(loader):
        batch_start_time = time.time()

        query = batch['query'].to(device, non_blocking=True)
        reference = batch['reference'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(query, reference)

            # Extract components for CEDLoss
            logits = outputs.get('logits', torch.zeros(query.size(0), device=device))
            query_cls_tokens = outputs.get('query_cls_tokens')
            ref_cls_tokens = outputs.get('ref_cls_tokens')
            patch_embeddings = outputs.get('patch_embeddings')

            # Compute CEDLoss if we have all components, otherwise fallback to BCE
            if query_cls_tokens is not None and ref_cls_tokens is not None and patch_embeddings is not None:
                # Average query CLS tokens across patches (6 patches per image)
                B = query.size(0)
                num_patches_per_image = query_cls_tokens.size(0) // B
                query_cls_tokens_avg = query_cls_tokens.reshape(B, num_patches_per_image, -1).mean(dim=1)

                # Average patch embeddings across patches
                patch_embeddings_avg = patch_embeddings.reshape(B, num_patches_per_image, -1).mean(dim=1)

                # Compute combined CED loss
                loss_dict = criterion(
                    cls_tokens_i=query_cls_tokens_avg,
                    cls_tokens_j=ref_cls_tokens,
                    patch_embeddings=patch_embeddings_avg,
                    labels=labels,
                    logits=logits.unsqueeze(1) if logits.dim() == 1 else logits
                )
                loss = loss_dict['total_loss']
            else:
                # Fallback to simple BCE loss
                loss = nn.BCEWithLogitsLoss()(logits, labels)

        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track batch time
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        # Track metrics
        if rank == 0 and batch_idx % 10 == 0:
            # Calculate ETA
            avg_batch_time = np.mean(batch_times[-50:])
            remaining_batches = len(loader) - batch_idx
            eta_seconds = remaining_batches * avg_batch_time
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            print(f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f} | Batch Time: {batch_time:.2f}s | ETA: {eta_str}")

        # Apply sigmoid to logits for metrics tracking
        tracker.update(torch.sigmoid(logits.detach()), labels, loss.item())

    if rank == 0:
        epoch_time = time.time() - epoch_start_time
        metrics = tracker.compute()
        tracker.print_metrics(f"Epoch {epoch} Train")
        print(f"Epoch {epoch} completed in {epoch_time/60:.2f} minutes")

        # Log metrics to file
        if log_file:
            log_entry = {
                'epoch': epoch,
                'phase': 'train',
                'loss': metrics.get('loss', 0.0),
                'accuracy': metrics.get('accuracy', 0.0),
                'epoch_time_minutes': epoch_time / 60,
                'timestamp': datetime.now().isoformat()
            }
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    return tracker.compute()


def main():
    parser = argparse.ArgumentParser(description='Train Baseline CEDetector (no APT)')

    # Data paths
    parser.add_argument('--disc21_path', type=str, default='/home/jowatson/Deep Learning/DISC21')
    parser.add_argument('--ndec_path', type=str, default='/home/jowatson/Deep Learning/NDEC')
    parser.add_argument('--hf_token_path', type=str, default='/home/jowatson/Deep Learning/Code/.env')

    # Training
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--ndec_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--subset_fraction', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # Model
    parser.add_argument('--dinov3_model', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--freeze_backbone', action='store_true')

    # Save/Load
    parser.add_argument('--save_dir', type=str, default='./checkpoints_baseline')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"Training Baseline CEDetector (no APT) on {world_size} GPUs")
        print(f"Arguments: {args}")
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        # Create metrics log file
        log_file = os.path.join(args.save_dir, 'training_metrics.jsonl')
        print(f"Logging metrics to: {log_file}")
    else:
        log_file = None

    # Create augmentations
    # Use 2-4 ops for faster initial convergence
    augmenter = CEDAugmentations(
        min_ops=2,
        max_ops=4,
        img_size=224
    )

    # Create data loaders for DISC21
    if rank == 0:
        print("\nLoading DISC21 dataset...")

    train_dataset = DISC21Dataset(
        root_dir=args.disc21_path,
        transform=augmenter,
        subset_fraction=args.subset_fraction
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model (BASELINE - no APT)
    if rank == 0:
        print("\nCreating Baseline CEDetector model (no APT)...")

    model = CEDetectorBaseline(
        dinov3_model=args.dinov3_model,
        hf_token_path=args.hf_token_path,
        freeze_backbone=args.freeze_backbone,
        num_query_patches=6,
        k_neighbors=10
    ).to(device)

    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=True,
        gradient_as_bucket_view=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = CEDLoss()

    # Mixed precision gradient scaler
    scaler = GradScaler() if args.use_amp else None

    if rank == 0:
        print(f"Mixed precision training: {'Enabled' if args.use_amp else 'Disabled'}")

    # Training loop - DISC21
    if rank == 0:
        print(f"\nTraining on DISC21 for {args.epochs} epochs...")
        training_start_time = time.time()

    best_mu_ap = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device, rank, scaler, log_file
        )

        scheduler.step()

        # Print overall progress and ETA
        if rank == 0:
            elapsed_time = time.time() - training_start_time
            epochs_completed = epoch + 1
            avg_epoch_time = elapsed_time / epochs_completed
            remaining_epochs = args.epochs - epochs_completed
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            total_eta = str(timedelta(seconds=int(elapsed_time + eta_seconds)))

            print(f"\n{'='*60}")
            print(f"Progress: {epochs_completed}/{args.epochs} epochs ({100*epochs_completed/args.epochs:.1f}%)")
            print(f"Elapsed: {str(timedelta(seconds=int(elapsed_time)))}")
            print(f"ETA: {eta_str}")
            print(f"Total estimated time: {total_eta}")
            print(f"{'='*60}\n")

        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            if train_metrics.get('mu_ap', 0) > best_mu_ap:
                best_mu_ap = train_metrics['mu_ap']
                best_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'mu_ap': best_mu_ap,
                }, best_path)
                print(f"Saved best model with µAP: {best_mu_ap:.4f}")

    # Final training on NDEC
    if rank == 0:
        print(f"\n\nFine-tuning on NDEC for {args.ndec_epochs} epochs...")

    ndec_dataset = NDECDataset(
        root_dir=args.ndec_path,
        transform=augmenter,
        subset_fraction=args.subset_fraction
    )

    ndec_sampler = DistributedSampler(
        ndec_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    ndec_loader = torch.utils.data.DataLoader(
        ndec_dataset,
        batch_size=args.batch_size,
        sampler=ndec_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    for epoch in range(args.ndec_epochs):
        ndec_sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, ndec_loader, criterion, optimizer,
            args.epochs + epoch, device, rank, scaler, log_file
        )

    # Final save
    if rank == 0:
        final_path = os.path.join(args.save_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'train_metrics': train_metrics,
        }, final_path)
        print(f"\nTraining complete! Saved final model: {final_path}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
