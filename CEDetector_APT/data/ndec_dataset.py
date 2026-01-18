"""
NDEC Dataset Loader
Negative Distractor for Edited Copy dataset
Focuses on hard negative distractor images
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import random


class NDECDataset(Dataset):
    """
    NDEC dataset for challenging copy-edit detection.
    Contains hard negative distractors that are visually similar but not copy-edits.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        subset_fraction: float = 1.0,
        include_negatives: bool = True,
    ):
        """
        Args:
            root_dir: Root directory of NDEC dataset
            transform: Augmentation transform
            subset_fraction: Fraction of dataset to use
            include_negatives: Whether to include hard negative pairs
        """
        self.root_dir = root_dir
        self.transform = transform
        self.include_negatives = include_negatives

        # Load ground truth
        gt_file = os.path.join(root_dir, 'public_ground_truth_h5.csv')
        if os.path.exists(gt_file):
            self.ground_truth = pd.read_csv(gt_file)
        else:
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

        # Query and reference directories
        self.query_dir = os.path.join(root_dir, 'query_set')
        self.reference_dir = os.path.join(root_dir, 'reference_set_true match')
        self.negative_dir = os.path.join(root_dir, 'negative_pair')

        print(f"NDEC Dataset:")
        print(f"  Total samples: {len(self.ground_truth)}")

        # Split into positives and negatives based on ground truth
        self.positive_pairs = []
        self.negative_pairs = []

        for idx, row in self.ground_truth.iterrows():
            query_id = row['query_id']
            ref_id = row['reference_id']

            # Query image path - NDEC has subdirectories
            query_path = self._find_query_path(query_id)

            if query_path is None:
                continue

            if pd.isna(ref_id) or ref_id == '':
                # This is a distractor (negative) query
                if include_negatives:
                    # Find corresponding hard negative
                    neg_path = self._find_negative_path(query_id)
                    if neg_path:
                        self.negative_pairs.append({
                            'query_id': query_id,
                            'query_path': query_path,
                            'reference_id': 'NEG',
                            'reference_path': neg_path,
                        })
            else:
                # Positive pair
                ref_path = os.path.join(self.reference_dir, f"{ref_id}.jpg")
                if not os.path.exists(ref_path):
                    ref_path = os.path.join(self.reference_dir, f"{ref_id}.png")

                if os.path.exists(ref_path):
                    self.positive_pairs.append({
                        'query_id': query_id,
                        'query_path': query_path,
                        'reference_id': ref_id,
                        'reference_path': ref_path,
                    })

        print(f"  Positive pairs: {len(self.positive_pairs)}")
        print(f"  Negative pairs: {len(self.negative_pairs)}")

        # Combine pairs
        self.all_pairs = self.positive_pairs + self.negative_pairs

        # Apply subset fraction
        if subset_fraction < 1.0:
            n_samples = int(len(self.all_pairs) * subset_fraction)
            indices = random.sample(range(len(self.all_pairs)), n_samples)
            self.all_pairs = [self.all_pairs[i] for i in indices]
            print(f"  Using {n_samples} samples ({subset_fraction*100:.1f}% of dataset)")

    def _find_query_path(self, query_id: str) -> Optional[str]:
        """Find query image path (may be in subdirectories)"""
        # NDEC query_set has subdirectories
        for subdir in os.listdir(self.query_dir):
            subdir_path = os.path.join(self.query_dir, subdir)
            if os.path.isdir(subdir_path):
                for ext in ['.jpg', '.png', '.jpeg']:
                    query_path = os.path.join(subdir_path, f"{query_id}{ext}")
                    if os.path.exists(query_path):
                        return query_path
        return None

    def _find_negative_path(self, query_id: str) -> Optional[str]:
        """Find corresponding hard negative image"""
        # Negative pairs are named similarly to query
        for ext in ['.jpg', '.png', '.jpeg']:
            neg_path = os.path.join(self.negative_dir, f"{query_id}{ext}")
            if os.path.exists(neg_path):
                return neg_path
        return None

    def __len__(self) -> int:
        return len(self.all_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
                - query: Query image tensor
                - reference: Reference image tensor
                - label: 1 if positive pair, 0 if negative
                - query_id: Query image ID
                - reference_id: Reference image ID
        """
        pair = self.all_pairs[idx]

        # Load query image
        query_img = Image.open(pair['query_path']).convert('RGB')

        # Load reference image
        ref_img = Image.open(pair['reference_path']).convert('RGB')

        # Determine label
        label = 1 if pair['reference_id'] != 'NEG' else 0

        # Apply transformations
        if self.transform is not None:
            query_img = self.transform(query_img)
            ref_img = self.transform(ref_img)
        else:
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            query_img = to_tensor(query_img)
            ref_img = to_tensor(ref_img)

        return {
            'query': query_img,
            'reference': ref_img,
            'label': torch.tensor(label, dtype=torch.float32),
            'query_id': pair['query_id'],
            'reference_id': pair['reference_id'],
        }

    def get_all_references(self) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Load all reference images for building retrieval corpus.

        Returns:
            Tuple of (reference_images, reference_ids)
        """
        ref_images = []
        ref_ids = []

        # Load from reference directory
        for filename in os.listdir(self.reference_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                ref_id = os.path.splitext(filename)[0]
                ref_path = os.path.join(self.reference_dir, filename)

                try:
                    img = Image.open(ref_path).convert('RGB')

                    if self.transform is not None:
                        img = self.transform(img)
                    else:
                        from torchvision import transforms
                        img = transforms.ToTensor()(img)

                    ref_images.append(img)
                    ref_ids.append(ref_id)

                except Exception as e:
                    print(f"Error loading {ref_path}: {e}")
                    continue

        return ref_images, ref_ids


def create_ndec_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_fraction: float = 1.0,
    transform=None,
) -> DataLoader:
    """
    Create NDEC data loader (typically used for final evaluation).

    Args:
        root_dir: Root directory of NDEC dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_fraction: Fraction of dataset to use
        transform: Augmentation transform

    Returns:
        NDEC data loader
    """
    dataset = NDECDataset(
        root_dir=root_dir,
        transform=transform,
        subset_fraction=subset_fraction,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
