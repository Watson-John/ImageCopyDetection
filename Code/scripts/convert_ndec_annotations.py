#!/usr/bin/env python3
"""
Convert NDEC challenge format to adaptive-dino-icd training format.

NDEC challenge format (public_ground_truth_h5.csv):
    query_id,reference_id
    Q25000,R352148        <- positive pair
    Q25001,               <- negative query (no match)

Adaptive-DINO-ICD format:
    img_a_path,img_b_path,label,direction,similar_pair_for_metric
    query_images/Q25000.jpg,reference_images/R352148.jpg,pos,a->b,true
    query_images/Q25001.jpg,negative_pair/00001/T613143.jpg,neg,a->b,false

This script creates a compatible annotation file without modifying the original.

Usage:
    python scripts/convert_ndec_annotations.py --ndec_root /path/to/NDEC --output pairs.csv
    python scripts/convert_ndec_annotations.py --ndec_root /path/to/NDEC --output pairs.csv --max_negatives 10000
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_negative_pair_images(negative_pair_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find all image pairs in negative_pair subdirectories.

    Each subdirectory contains an original image and a transformed version.
    Returns list of (original, transformed) tuples.
    """
    pairs = []

    if not negative_pair_dir.exists():
        logger.warning(f"Negative pair directory not found: {negative_pair_dir}")
        return pairs

    for subdir in sorted(negative_pair_dir.iterdir()):
        if not subdir.is_dir():
            continue

        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))

        if len(images) >= 2:
            # T* prefix indicates transformed, others are originals
            transformed = [img for img in images if img.name.startswith("T")]
            originals = [img for img in images if not img.name.startswith("T")]

            if transformed and originals:
                pairs.append((originals[0], transformed[0]))

    return pairs


def convert_ndec_annotations(
    ndec_root: Path,
    output_file: Path,
    max_negatives: int = None,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convert NDEC annotations to training format.

    Args:
        ndec_root: Root directory of NDEC dataset
        output_file: Output CSV file path
        max_negatives: Maximum number of negative pairs to include (None = all)
        negative_ratio: Ratio of negatives to positives (default 1.0 = equal)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with converted annotations
    """
    random.seed(seed)

    # Define paths
    groundtruth_file = ndec_root / "public_ground_truth_h5.csv"
    query_dir = ndec_root / "query_set" / "query_images_h5"
    reference_dir = ndec_root / "reference_set_true match"
    negative_pair_dir = ndec_root / "negative_pair"

    # Validate paths
    if not groundtruth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {groundtruth_file}")

    logger.info(f"Loading ground truth from: {groundtruth_file}")
    df_gt = pd.read_csv(groundtruth_file)

    logger.info(f"Query images directory: {query_dir}")
    logger.info(f"Reference images directory: {reference_dir}")
    logger.info(f"Negative pair directory: {negative_pair_dir}")

    # Separate positive and negative queries
    df_positive = df_gt[df_gt["reference_id"].notna()].copy()
    df_negative = df_gt[df_gt["reference_id"].isna()].copy()

    logger.info(f"Found {len(df_positive)} positive pairs")
    logger.info(f"Found {len(df_negative)} negative queries")

    # Build positive pairs
    positive_pairs = []
    missing_queries = 0
    missing_refs = 0

    for _, row in df_positive.iterrows():
        query_id = row["query_id"]
        ref_id = row["reference_id"]

        # Build paths (relative to image_root which will be ndec_root)
        query_path = Path("query_set") / "query_images_h5" / f"{query_id}.jpg"
        ref_path = Path("reference_set_true match") / f"{ref_id}.jpg"

        # Verify files exist
        full_query_path = ndec_root / query_path
        full_ref_path = ndec_root / ref_path

        if not full_query_path.exists():
            missing_queries += 1
            continue
        if not full_ref_path.exists():
            missing_refs += 1
            continue

        positive_pairs.append({
            "img_a_path": str(query_path),
            "img_b_path": str(ref_path),
            "label": "pos",
            "direction": "a->b",  # query -> reference
            "similar_pair_for_metric": True,
        })

    if missing_queries > 0:
        logger.warning(f"Missing query images: {missing_queries}")
    if missing_refs > 0:
        logger.warning(f"Missing reference images: {missing_refs}")

    logger.info(f"Created {len(positive_pairs)} positive pairs")

    # Build negative pairs from negative_pair directory
    logger.info("Scanning negative_pair directory for distractor pairs...")
    negative_image_pairs = find_negative_pair_images(negative_pair_dir)
    logger.info(f"Found {len(negative_image_pairs)} negative image pairs")

    # Determine how many negatives to include
    target_negatives = int(len(positive_pairs) * negative_ratio)
    if max_negatives is not None:
        target_negatives = min(target_negatives, max_negatives)
    target_negatives = min(target_negatives, len(negative_image_pairs))

    logger.info(f"Selecting {target_negatives} negative pairs")

    # Randomly sample negative pairs
    if target_negatives < len(negative_image_pairs):
        selected_negatives = random.sample(negative_image_pairs, target_negatives)
    else:
        selected_negatives = negative_image_pairs

    negative_pairs = []
    for orig_path, trans_path in selected_negatives:
        # Make paths relative to ndec_root
        rel_orig = orig_path.relative_to(ndec_root)
        rel_trans = trans_path.relative_to(ndec_root)

        negative_pairs.append({
            "img_a_path": str(rel_trans),  # transformed as query
            "img_b_path": str(rel_orig),   # original as reference
            "label": "neg",
            "direction": "a->b",
            "similar_pair_for_metric": False,
        })

    logger.info(f"Created {len(negative_pairs)} negative pairs")

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Create DataFrame
    df_output = pd.DataFrame(all_pairs)

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_file, index=False)

    logger.info(f"Saved {len(df_output)} pairs to: {output_file}")
    logger.info(f"  - Positive: {len(positive_pairs)}")
    logger.info(f"  - Negative: {len(negative_pairs)}")

    return df_output


def main():
    parser = argparse.ArgumentParser(
        description="Convert NDEC annotations to adaptive-dino-icd format"
    )
    parser.add_argument(
        "--ndec_root",
        type=str,
        default="/home/jowatson/Deep Learning/NDEC",
        help="Root directory of NDEC dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: {ndec_root}/pairs_converted.csv)",
    )
    parser.add_argument(
        "--max_negatives",
        type=int,
        default=None,
        help="Maximum number of negative pairs to include",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive pairs (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    ndec_root = Path(args.ndec_root)

    if args.output is None:
        output_file = ndec_root / "pairs_converted.csv"
    else:
        output_file = Path(args.output)

    convert_ndec_annotations(
        ndec_root=ndec_root,
        output_file=output_file,
        max_negatives=args.max_negatives,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
