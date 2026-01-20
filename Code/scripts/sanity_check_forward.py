#!/usr/bin/env python3
"""
Sanity check script for verifying forward pass through the model.

Runs a forward pass on random data and prints output shapes.
Uses offline stub by default (no network required).

Run with:
    python scripts/sanity_check_forward.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sanity check for model forward pass")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to offline stub)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for test",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use_adaptive",
        action="store_true",
        default=True,
        help="Use adaptive patch tokenization",
    )
    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disable adaptive patch tokenization",
    )

    args = parser.parse_args()

    if args.no_adaptive:
        args.use_adaptive = False

    logger.info("=" * 60)
    logger.info("Adaptive-DINO-ICD Sanity Check")
    logger.info("=" * 60)

    # Import modules
    try:
        from adaptive_dino_icd.utils.config import Config, BackboneConfig, APTConfig
        from adaptive_dino_icd.backbone import AdaptiveBackbone
        from adaptive_dino_icd.losses import ASLLossModule
        logger.info("Successfully imported all modules")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you've installed the package or added src to PYTHONPATH")
        sys.exit(1)

    # Create config
    if args.config:
        from adaptive_dino_icd.utils.config import load_config
        config = load_config(args.config)
        backbone_config = config.backbone
        apt_config = config.apt
    else:
        # Use offline stub by default
        backbone_config = BackboneConfig(
            offline_stub=True,
            embed_dim=768,
        )
        apt_config = APTConfig()

    logger.info(f"Using device: {args.device}")
    logger.info(f"Offline stub: {backbone_config.offline_stub}")
    logger.info(f"Adaptive patches: {args.use_adaptive}")

    # Create model
    logger.info("\nCreating model...")
    try:
        model = AdaptiveBackbone(
            backbone_config=backbone_config,
            apt_config=apt_config,
            use_adaptive_patches=args.use_adaptive,
        )
        model = model.to(args.device)
        model.eval()
        logger.info(f"Model created successfully")
        logger.info(f"  - Embedding dimension: {model.embed_dim}")
        logger.info(f"  - Total parameters: {model.get_num_params(trainable_only=False):,}")
        logger.info(f"  - Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create dummy input
    logger.info(f"\nCreating dummy input (B={args.batch_size}, size={args.image_size})...")
    dummy_input = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size,
        device=args.device,
    )
    logger.info(f"Input shape: {dummy_input.shape}")

    # Forward pass
    logger.info("\nRunning forward pass...")
    try:
        with torch.no_grad():
            outputs = model(dummy_input, return_entropy=True)

        logger.info("Forward pass successful!")
        logger.info("\nOutput shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  - {key}: {value.shape}")
            elif isinstance(value, dict):
                logger.info(f"  - {key}: dict with {len(value)} entries")
            elif isinstance(value, list):
                logger.info(f"  - {key}: list with {len(value)} entries")
            else:
                logger.info(f"  - {key}: {type(value)}")

    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test loss computation
    logger.info("\n" + "=" * 60)
    logger.info("Testing ASL Loss Module")
    logger.info("=" * 60)

    try:
        loss_module = ASLLossModule(lambda_mtr=0.5)
        logger.info("Loss module created successfully")

        # Create dummy features (simulating ref and query)
        feat_ref = outputs["global_feats"]
        feat_query = torch.randn_like(feat_ref)  # Different features

        # Dummy labels
        is_copy = torch.tensor([True, False][:args.batch_size], device=args.device)
        stream = ["disc"] * args.batch_size
        direction = ["full->crop"] * args.batch_size

        loss, info = loss_module(
            feat_ref, feat_query, is_copy, stream, direction
        )

        logger.info(f"Loss computation successful!")
        logger.info(f"  - Total loss: {loss.item():.4f}")
        logger.info(f"  - Norm ratio loss: {info['norm_ratio_loss']:.4f}")
        logger.info(f"  - Metric loss: {info['metric_loss']:.4f}")

    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test inference matcher
    logger.info("\n" + "=" * 60)
    logger.info("Testing D2LV Matcher")
    logger.info("=" * 60)

    try:
        from adaptive_dino_icd.inference import D2LVMatcher

        matcher = D2LVMatcher(global_weight=0.5, local_weight=0.5)
        logger.info("Matcher created successfully")

        # Create dummy features for matching
        query_feats = {
            "global_feats": outputs["global_feats"],
            "local_feats": outputs["local_feats"],
            "local_coords": outputs["local_coords"],
            "attention_mask": outputs["attention_mask"],
        }

        gallery_feats = {
            "global_feats": torch.randn_like(outputs["global_feats"]),
            "local_feats": torch.randn_like(outputs["local_feats"]),
            "local_coords": outputs["local_coords"],
            "attention_mask": outputs["attention_mask"],
        }

        scores = matcher(query_feats, gallery_feats)
        logger.info(f"Matching successful!")
        logger.info(f"  - Scores shape: {scores.shape}")
        logger.info(f"  - Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    except Exception as e:
        logger.error(f"Matching failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK PASSED!")
    logger.info("=" * 60)
    logger.info("All components working correctly:")
    logger.info("  - AdaptiveBackbone: OK")
    logger.info("  - ASLLossModule: OK")
    logger.info("  - D2LVMatcher: OK")


if __name__ == "__main__":
    main()
