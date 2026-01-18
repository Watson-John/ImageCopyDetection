"""
DISC21 Dataset Loader
Image Similarity Challenge 2021 dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import random


class DISC21Dataset(Dataset):
    """
    DISC21 dataset for image copy detection.
    Contains 1M reference images and query images with ground truth.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        subset_fraction: float = 1.0,
        max_references: Optional[int] = None,
        return_pairs: bool = True,
    ):
        """
        Args:
            root_dir: Root directory of DISC21 dataset
            split: 'train' or 'val'
            transform: Augmentation transform
            subset_fraction: Fraction of dataset to use (for faster training)
            max_references: Maximum number of reference images to load
            return_pairs: Whether to return query-reference pairs
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_pairs = return_pairs

        # Load ground truth
        gt_file = os.path.join(root_dir, 'dev_queries_groundtruth.csv')
        if os.path.exists(gt_file):
            self.ground_truth = pd.read_csv(gt_file, header=None, names=['query_id', 'reference_id'])
        else:
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

        # Query images directory
        self.query_dir = os.path.join(root_dir, 'dev_queries_50k_0')

        # Reference images directories
        self.reference_dirs = []
        for i in range(20):  # DISC21 has refs_50k_0 to refs_50k_19
            ref_dir = os.path.join(root_dir, f'refs_50k_{i}')
            if os.path.exists(ref_dir):
                self.reference_dirs.append(ref_dir)

        print(f"Found {len(self.reference_dirs)} reference directories")

        # Build reference image index
        self.reference_index = {}
        total_refs = 0

        for ref_dir in self.reference_dirs:
            ref_files = [f for f in os.listdir(ref_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for ref_file in ref_files:
                # Extract reference ID from filename (e.g., "R000123.jpg" -> "R000123")
                ref_id = os.path.splitext(ref_file)[0]
                self.reference_index[ref_id] = os.path.join(ref_dir, ref_file)
                total_refs += 1

                if max_references is not None and total_refs >= max_references:
                    break

            if max_references is not None and total_refs >= max_references:
                break

        print(f"Indexed {len(self.reference_index)} reference images")

        # Apply subset fraction
        if subset_fraction < 1.0:
            n_samples = int(len(self.ground_truth) * subset_fraction)
            self.ground_truth = self.ground_truth.sample(n=n_samples, random_state=42).reset_index(drop=True)
            print(f"Using {n_samples} samples ({subset_fraction*100:.1f}% of dataset)")

        # Filter ground truth to only include pairs where reference exists
        self.valid_pairs = []
        for idx, row in self.ground_truth.iterrows():
            query_id = row['query_id']
            ref_id = row['reference_id']

            query_path = os.path.join(self.query_dir, f"{query_id}.jpg")
            if ref_id in self.reference_index and os.path.exists(query_path):
                self.valid_pairs.append({
                    'query_id': query_id,
                    'query_path': query_path,
                    'reference_id': ref_id,
                    'reference_path': self.reference_index[ref_id]
                })

        print(f"Found {len(self.valid_pairs)} valid query-reference pairs")

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
                - query: Query image tensor
                - reference: Reference image tensor (positive or negative)
                - label: 1 if positive pair, 0 if negative
                - query_id: Query image ID
                - reference_id: Reference image ID
        """
        # Get positive pair
        pair = self.valid_pairs[idx]

        # Load query image
        query_img = Image.open(pair['query_path']).convert('RGB')

        # With 50% probability, create negative pair
        if self.return_pairs and random.random() > 0.5:
            # Negative pair: different reference image
            neg_idx = random.randint(0, len(self.valid_pairs) - 1)
            while neg_idx == idx:
                neg_idx = random.randint(0, len(self.valid_pairs) - 1)

            ref_path = self.valid_pairs[neg_idx]['reference_path']
            ref_id = self.valid_pairs[neg_idx]['reference_id']
            label = 0
        else:
            # Positive pair: correct reference
            ref_path = pair['reference_path']
            ref_id = pair['reference_id']
            label = 1

        # Load reference image
        ref_img = Image.open(ref_path).convert('RGB')

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
            'reference_id': ref_id,
        }

    def get_all_references(self, max_refs: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Load all reference images for building retrieval corpus.

        Args:
            max_refs: Maximum number of references to load

        Returns:
            Tuple of (reference_images, reference_ids)
        """
        ref_images = []
        ref_ids = []

        count = 0
        for ref_id, ref_path in self.reference_index.items():
            try:
                img = Image.open(ref_path).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)

                ref_images.append(img)
                ref_ids.append(ref_id)

                count += 1
                if max_refs is not None and count >= max_refs:
                    break

            except Exception as e:
                print(f"Error loading {ref_path}: {e}")
                continue

        return ref_images, ref_ids


def create_disc21_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_fraction: float = 1.0,
    max_references: Optional[int] = None,
    transform=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DISC21 data loaders.

    Args:
        root_dir: Root directory of DISC21 dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_fraction: Fraction of dataset to use
        max_references: Maximum number of reference images
        transform: Augmentation transform

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # For DISC21, we'll use a 90/10 train/val split
    full_dataset = DISC21Dataset(
        root_dir=root_dir,
        transform=transform,
        subset_fraction=subset_fraction,
        max_references=max_references,
    )

    # Split into train and val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
