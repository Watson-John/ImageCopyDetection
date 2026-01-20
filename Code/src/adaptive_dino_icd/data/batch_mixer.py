"""
Batch Mixer for fixed-ratio DISC/NDEC training.

Maintains exact 70% DISC : 30% NDEC sampling ratio every training step.
No warmup - this ratio is used from step 0 across all epochs.
"""

import logging
import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


class MixedBatchSampler(Sampler):
    """
    Sampler that yields batches with fixed DISC/NDEC ratio.

    Ensures exactly disc_ratio of each batch comes from DISC dataset
    and (1 - disc_ratio) comes from NDEC dataset.

    Args:
        disc_dataset: DISC dataset
        ndec_dataset: NDEC dataset
        batch_size: Total batch size
        disc_ratio: Ratio of DISC samples per batch (default: 0.7)
        seed: Random seed for reproducibility
        drop_last: Whether to drop incomplete final batch

    Example:
        sampler = MixedBatchSampler(
            disc_dataset, ndec_dataset,
            batch_size=32, disc_ratio=0.7
        )
        # Each batch will have 22 DISC samples + 10 NDEC samples
    """

    def __init__(
        self,
        disc_dataset: Dataset,
        ndec_dataset: Dataset,
        batch_size: int,
        disc_ratio: float = 0.7,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.disc_dataset = disc_dataset
        self.ndec_dataset = ndec_dataset
        self.batch_size = batch_size
        self.disc_ratio = disc_ratio
        self.seed = seed
        self.drop_last = drop_last

        # Calculate samples per stream per batch
        self.disc_per_batch = int(round(batch_size * disc_ratio))
        self.ndec_per_batch = batch_size - self.disc_per_batch

        # Validate
        if self.disc_per_batch == 0:
            logger.warning("disc_per_batch is 0, adjusting to at least 1")
            self.disc_per_batch = 1
            self.ndec_per_batch = batch_size - 1

        if self.ndec_per_batch == 0:
            logger.warning("ndec_per_batch is 0, adjusting to at least 1")
            self.ndec_per_batch = 1
            self.disc_per_batch = batch_size - 1

        # Calculate number of complete batches
        disc_batches = len(disc_dataset) // self.disc_per_batch
        ndec_batches = len(ndec_dataset) // self.ndec_per_batch
        self.num_batches = min(disc_batches, ndec_batches)

        if not drop_last:
            # Add one more batch if there are remaining samples
            disc_remain = len(disc_dataset) % self.disc_per_batch
            ndec_remain = len(ndec_dataset) % self.ndec_per_batch
            if disc_remain > 0 or ndec_remain > 0:
                self.num_batches += 1

        logger.info(
            f"MixedBatchSampler: batch_size={batch_size}, "
            f"disc_per_batch={self.disc_per_batch}, ndec_per_batch={self.ndec_per_batch}, "
            f"num_batches={self.num_batches}"
        )

    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        """
        Yield batches of (stream, index) tuples.

        Each batch contains disc_per_batch items from "disc" stream
        and ndec_per_batch items from "ndec" stream.
        """
        # Create shuffled indices for each dataset
        g = torch.Generator()
        g.manual_seed(self.seed)

        disc_indices = torch.randperm(len(self.disc_dataset), generator=g).tolist()
        ndec_indices = torch.randperm(len(self.ndec_dataset), generator=g).tolist()

        disc_ptr = 0
        ndec_ptr = 0

        for _ in range(self.num_batches):
            batch = []

            # Add DISC samples
            for _ in range(self.disc_per_batch):
                if disc_ptr >= len(disc_indices):
                    # Reshuffle if exhausted
                    disc_indices = torch.randperm(len(self.disc_dataset), generator=g).tolist()
                    disc_ptr = 0
                batch.append(("disc", disc_indices[disc_ptr]))
                disc_ptr += 1

            # Add NDEC samples
            for _ in range(self.ndec_per_batch):
                if ndec_ptr >= len(ndec_indices):
                    # Reshuffle if exhausted
                    ndec_indices = torch.randperm(len(self.ndec_dataset), generator=g).tolist()
                    ndec_ptr = 0
                batch.append(("ndec", ndec_indices[ndec_ptr]))
                ndec_ptr += 1

            yield batch

    def __len__(self) -> int:
        return self.num_batches


class BatchMixer:
    """
    High-level batch mixer combining DISC and NDEC data streams.

    Creates dataloaders and provides iteration over mixed batches.
    Maintains exact ratio every batch, not just on average.

    Args:
        disc_dataset: DISC dataset instance
        ndec_dataset: NDEC dataset instance
        batch_size: Total batch size
        disc_ratio: Ratio of DISC samples (default: 0.7)
        num_workers: DataLoader workers
        seed: Random seed
        pin_memory: Pin memory for GPU transfer

    Example:
        mixer = BatchMixer(disc_dataset, ndec_dataset, batch_size=32)
        for batch in mixer:
            # batch contains mixed DISC and NDEC samples
            disc_samples = [s for s in batch if s["stream"] == "disc"]
            ndec_samples = [s for s in batch if s["stream"] == "ndec"]
    """

    def __init__(
        self,
        disc_dataset: Dataset,
        ndec_dataset: Dataset,
        batch_size: int = 32,
        disc_ratio: float = 0.7,
        num_workers: int = 4,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        self.disc_dataset = disc_dataset
        self.ndec_dataset = ndec_dataset
        self.batch_size = batch_size
        self.disc_ratio = disc_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory

        # Calculate per-stream counts
        self.disc_per_batch = int(round(batch_size * disc_ratio))
        self.ndec_per_batch = batch_size - self.disc_per_batch

        # Create separate dataloaders
        self._create_dataloaders()

        logger.info(
            f"BatchMixer initialized: {self.disc_per_batch} DISC + "
            f"{self.ndec_per_batch} NDEC per batch"
        )

    def _create_dataloaders(self) -> None:
        """Create dataloaders for each stream."""
        # Use infinite sampling for seamless epoch boundaries
        self.disc_loader = DataLoader(
            self.disc_dataset,
            batch_size=self.disc_per_batch,
            shuffle=True,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.ndec_loader = DataLoader(
            self.ndec_dataset,
            batch_size=self.ndec_per_batch,
            shuffle=True,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.seed + 1),
        )

    def __iter__(self) -> Iterator[Dict[str, List]]:
        """Iterate over mixed batches."""
        disc_iter = iter(self.disc_loader)
        ndec_iter = iter(self.ndec_loader)

        # Calculate number of batches based on smaller dataset
        num_batches = min(len(self.disc_loader), len(self.ndec_loader))

        for _ in range(num_batches):
            try:
                disc_batch = next(disc_iter)
            except StopIteration:
                disc_iter = iter(self.disc_loader)
                disc_batch = next(disc_iter)

            try:
                ndec_batch = next(ndec_iter)
            except StopIteration:
                ndec_iter = iter(self.ndec_loader)
                ndec_batch = next(ndec_iter)

            # Combine batches
            yield self._merge_batches(disc_batch, ndec_batch)

    def _merge_batches(
        self,
        disc_batch: Dict[str, torch.Tensor],
        ndec_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Merge DISC and NDEC batches into single batch.

        Returns a dict with:
        - img_ref: Reference images (full for DISC, img_a or img_b for NDEC based on direction)
        - img_query: Query images (crop for DISC, the other image for NDEC)
        - stream: List of stream labels ("disc" or "ndec")
        - direction: List of direction strings
        - is_copy: Tensor of copy labels
        - label: List of labels (for NDEC)
        - similar_pair_for_metric: List of bools (for NDEC)
        """
        # DISC: full is reference, crop is query
        disc_ref = disc_batch["img_full"]  # (N_disc, C, H, W)
        disc_query = disc_batch["img_crop"]

        # NDEC: determine ref/query based on direction
        ndec_ref = []
        ndec_query = []
        for i in range(ndec_batch["img_a"].shape[0]):
            direction = ndec_batch["direction"][i]
            if direction == "a->b":
                ndec_ref.append(ndec_batch["img_a"][i])
                ndec_query.append(ndec_batch["img_b"][i])
            else:  # "b->a"
                ndec_ref.append(ndec_batch["img_b"][i])
                ndec_query.append(ndec_batch["img_a"][i])

        ndec_ref = torch.stack(ndec_ref, dim=0) if ndec_ref else torch.zeros(0, *disc_ref.shape[1:])
        ndec_query = torch.stack(ndec_query, dim=0) if ndec_query else torch.zeros(0, *disc_ref.shape[1:])

        # Concatenate
        img_ref = torch.cat([disc_ref, ndec_ref], dim=0)
        img_query = torch.cat([disc_query, ndec_query], dim=0)

        # Build stream labels
        n_disc = disc_ref.shape[0]
        n_ndec = ndec_ref.shape[0]

        streams = ["disc"] * n_disc + ["ndec"] * n_ndec
        directions = list(disc_batch["direction"]) + list(ndec_batch["direction"])

        # is_copy: True for DISC (always positive) and NDEC positives
        disc_is_copy = disc_batch["is_copy"]  # Should be all True
        ndec_is_copy = ndec_batch["is_copy"]
        is_copy = torch.cat([disc_is_copy, ndec_is_copy], dim=0)

        # NDEC-specific fields
        labels = ["pos"] * n_disc + list(ndec_batch["label"])
        similar_for_metric = [True] * n_disc + list(ndec_batch["similar_pair_for_metric"])

        return {
            "img_ref": img_ref,
            "img_query": img_query,
            "stream": streams,
            "direction": directions,
            "is_copy": is_copy,
            "label": labels,
            "similar_pair_for_metric": similar_for_metric,
        }

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return min(len(self.disc_loader), len(self.ndec_loader))

    def get_ratio_stats(self) -> Dict[str, float]:
        """Get actual ratio statistics."""
        return {
            "disc_ratio": self.disc_per_batch / self.batch_size,
            "ndec_ratio": self.ndec_per_batch / self.batch_size,
            "disc_per_batch": self.disc_per_batch,
            "ndec_per_batch": self.ndec_per_batch,
            "batch_size": self.batch_size,
        }
