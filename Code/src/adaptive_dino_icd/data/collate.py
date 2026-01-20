"""
Collate functions for mixed DISC/NDEC batches.

Handles combining samples from different streams into unified batch format.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch


@dataclass
class PairBatch:
    """
    Structured batch of image pairs for training.

    Attributes:
        img_ref: Reference images (B, C, H, W)
        img_query: Query images (B, C, H, W)
        stream: Stream labels ("disc" or "ndec") per sample
        direction: Direction flags per sample
        is_copy: Boolean tensor indicating positive pairs
        label: Label strings for NDEC samples
        similar_pair_for_metric: Boolean for metric loss behavior
        batch_size: Total batch size
        disc_count: Number of DISC samples
        ndec_count: Number of NDEC samples
    """

    img_ref: torch.Tensor
    img_query: torch.Tensor
    stream: List[str]
    direction: List[str]
    is_copy: torch.Tensor
    label: List[str]
    similar_pair_for_metric: List[bool]

    @property
    def batch_size(self) -> int:
        return self.img_ref.shape[0]

    @property
    def disc_count(self) -> int:
        return sum(1 for s in self.stream if s == "disc")

    @property
    def ndec_count(self) -> int:
        return sum(1 for s in self.stream if s == "ndec")

    @property
    def disc_mask(self) -> torch.Tensor:
        """Boolean mask for DISC samples."""
        return torch.tensor([s == "disc" for s in self.stream])

    @property
    def ndec_mask(self) -> torch.Tensor:
        """Boolean mask for NDEC samples."""
        return torch.tensor([s == "ndec" for s in self.stream])

    def to(self, device: torch.device) -> "PairBatch":
        """Move tensors to device."""
        return PairBatch(
            img_ref=self.img_ref.to(device),
            img_query=self.img_query.to(device),
            stream=self.stream,
            direction=self.direction,
            is_copy=self.is_copy.to(device),
            label=self.label,
            similar_pair_for_metric=self.similar_pair_for_metric,
        )


def mixed_collate_fn(
    samples: List[Dict],
    image_size: Optional[int] = None,
) -> PairBatch:
    """
    Collate function for mixed DISC/NDEC batches.

    Handles samples from both DISCDataset and NDECDataset,
    combining them into a unified PairBatch structure.

    Args:
        samples: List of sample dictionaries from datasets
        image_size: Optional target image size for resizing

    Returns:
        PairBatch with all samples combined
    """
    img_refs = []
    img_queries = []
    streams = []
    directions = []
    is_copies = []
    labels = []
    similar_for_metrics = []

    for sample in samples:
        stream = sample["stream"]
        streams.append(stream)

        if stream == "disc":
            # DISC sample: full is ref, crop is query
            img_refs.append(sample["img_full"])
            img_queries.append(sample["img_crop"])
            directions.append(sample["direction"])
            is_copies.append(sample["is_copy"])
            labels.append("pos")  # DISC is always positive
            similar_for_metrics.append(True)

        elif stream == "ndec":
            # NDEC sample: determine ref/query from direction
            direction = sample["direction"]
            directions.append(direction)

            if direction == "a->b":
                img_refs.append(sample["img_a"])
                img_queries.append(sample["img_b"])
            else:  # "b->a"
                img_refs.append(sample["img_b"])
                img_queries.append(sample["img_a"])

            is_copies.append(sample["is_copy"])
            labels.append(sample["label"])
            similar_for_metrics.append(sample.get("similar_pair_for_metric", False))

        else:
            raise ValueError(f"Unknown stream: {stream}")

    # Stack tensors
    img_ref = torch.stack(img_refs, dim=0)
    img_query = torch.stack(img_queries, dim=0)

    # Handle is_copy - could be bool or tensor
    if isinstance(is_copies[0], bool):
        is_copy = torch.tensor(is_copies, dtype=torch.bool)
    else:
        is_copy = torch.stack([torch.tensor(ic) for ic in is_copies])

    # Resize if needed
    if image_size is not None:
        img_ref = _resize_batch(img_ref, image_size)
        img_query = _resize_batch(img_query, image_size)

    return PairBatch(
        img_ref=img_ref,
        img_query=img_query,
        stream=streams,
        direction=directions,
        is_copy=is_copy,
        label=labels,
        similar_pair_for_metric=similar_for_metrics,
    )


def _resize_batch(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize batch of images to target size."""
    import torch.nn.functional as F

    if images.shape[-1] == size and images.shape[-2] == size:
        return images

    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def disc_collate_fn(samples: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for DISC-only batches.

    Args:
        samples: List of DISC samples

    Returns:
        Dictionary batch
    """
    img_fulls = torch.stack([s["img_full"] for s in samples])
    img_crops = torch.stack([s["img_crop"] for s in samples])
    is_copies = torch.tensor([s["is_copy"] for s in samples])

    return {
        "img_full": img_fulls,
        "img_crop": img_crops,
        "stream": [s["stream"] for s in samples],
        "direction": [s["direction"] for s in samples],
        "is_copy": is_copies,
        "path": [s["path"] for s in samples],
    }


def ndec_collate_fn(samples: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for NDEC-only batches.

    Args:
        samples: List of NDEC samples

    Returns:
        Dictionary batch
    """
    img_as = torch.stack([s["img_a"] for s in samples])
    img_bs = torch.stack([s["img_b"] for s in samples])
    is_copies = torch.tensor([s["is_copy"] for s in samples])

    return {
        "img_a": img_as,
        "img_b": img_bs,
        "stream": [s["stream"] for s in samples],
        "label": [s["label"] for s in samples],
        "direction": [s["direction"] for s in samples],
        "is_copy": is_copies,
        "similar_pair_for_metric": [s["similar_pair_for_metric"] for s in samples],
    }
