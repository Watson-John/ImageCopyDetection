"""
NDEC Dataset for supervised copy detection training.

Loads labeled image pairs with direction flags for asymmetric similarity learning.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class NDECDataset(Dataset):
    """
    NDEC-style dataset for supervised image copy detection.

    Loads labeled pairs from CSV/JSON with:
    - img_a_path, img_b_path: Paths to the two images
    - label: "pos" (positive/copy) or "neg" (negative/different)
    - direction: Which image is reference (e.g., "a->b" means a is reference)
    - similar_pair_for_metric: Optional bool for metric loss behavior

    Args:
        annotation_file: Path to CSV or JSON annotation file
        image_root: Root directory for image paths (prepended to paths in annotations)
        transform: Transform to apply to both images
        max_samples: Maximum number of samples (None for all)

    Annotation file format (CSV):
        img_a_path,img_b_path,label,direction,similar_pair_for_metric
        images/001.jpg,images/002.jpg,pos,a->b,true
        images/003.jpg,images/004.jpg,neg,a->b,false

    Annotation file format (JSON):
        [
            {
                "img_a_path": "images/001.jpg",
                "img_b_path": "images/002.jpg",
                "label": "pos",
                "direction": "a->b",
                "similar_pair_for_metric": true
            },
            ...
        ]

    Yields per __getitem__:
        Dictionary with:
            - img_a: Image A tensor
            - img_b: Image B tensor
            - stream: "ndec"
            - label: "pos" or "neg"
            - direction: "a->b" or "b->a"
            - similar_pair_for_metric: bool
            - is_copy: bool (True for positive pairs)
    """

    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_root: Union[str, Path],
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.transform = transform

        # Load annotations
        self.annotations = self._load_annotations()

        if max_samples is not None and len(self.annotations) > max_samples:
            self.annotations = self.annotations[:max_samples]

        logger.info(
            f"NDECDataset: Loaded {len(self.annotations)} pairs from {annotation_file}"
        )

    def _load_annotations(self) -> List[Dict]:
        """Load annotations from CSV or JSON file."""
        if not self.annotation_file.exists():
            logger.warning(f"NDEC annotation file does not exist: {self.annotation_file}")
            return []

        suffix = self.annotation_file.suffix.lower()

        if suffix == ".csv":
            return self._load_csv()
        elif suffix == ".json":
            return self._load_json()
        else:
            raise ValueError(f"Unsupported annotation file format: {suffix}")

    def _load_csv(self) -> List[Dict]:
        """Load annotations from CSV."""
        df = pd.read_csv(self.annotation_file)

        # Required columns
        required = ["img_a_path", "img_b_path", "label", "direction"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert to list of dicts
        annotations = []
        for _, row in df.iterrows():
            ann = {
                "img_a_path": row["img_a_path"],
                "img_b_path": row["img_b_path"],
                "label": row["label"],
                "direction": row["direction"],
                "similar_pair_for_metric": row.get("similar_pair_for_metric", False),
            }

            # Handle various boolean representations
            if isinstance(ann["similar_pair_for_metric"], str):
                ann["similar_pair_for_metric"] = ann["similar_pair_for_metric"].lower() in [
                    "true", "1", "yes"
                ]

            annotations.append(ann)

        return annotations

    def _load_json(self) -> List[Dict]:
        """Load annotations from JSON."""
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            annotations = data
        elif isinstance(data, dict) and "pairs" in data:
            annotations = data["pairs"]
        else:
            raise ValueError("JSON must be a list or dict with 'pairs' key")

        # Ensure all required fields
        for ann in annotations:
            if "similar_pair_for_metric" not in ann:
                ann["similar_pair_for_metric"] = False

        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """Get a labeled pair."""
        ann = self.annotations[idx]

        # Build full paths
        path_a = self.image_root / ann["img_a_path"]
        path_b = self.image_root / ann["img_b_path"]

        # Load images
        try:
            img_a = Image.open(path_a).convert("RGB")
            img_b = Image.open(path_b).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load images: {path_a}, {path_b}: {e}")
            return self._get_dummy_sample(ann)

        # Apply transforms
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        else:
            img_a = self._default_transform(img_a)
            img_b = self._default_transform(img_b)

        return {
            "img_a": img_a,
            "img_b": img_b,
            "stream": "ndec",
            "label": ann["label"],
            "direction": ann["direction"],
            "similar_pair_for_metric": bool(ann["similar_pair_for_metric"]),
            "is_copy": ann["label"] == "pos",
        }

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default transform: resize and convert to tensor."""
        import torchvision.transforms.functional as TF

        image = image.resize((224, 224), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        return tensor

    def _get_dummy_sample(self, ann: Dict) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """Return dummy sample for error cases."""
        dummy = torch.zeros(3, 224, 224)
        return {
            "img_a": dummy,
            "img_b": dummy,
            "stream": "ndec",
            "label": ann.get("label", "neg"),
            "direction": ann.get("direction", "a->b"),
            "similar_pair_for_metric": False,
            "is_copy": False,
        }

    def get_statistics(self) -> Dict[str, int]:
        """Get dataset statistics."""
        pos_count = sum(1 for a in self.annotations if a["label"] == "pos")
        neg_count = len(self.annotations) - pos_count

        return {
            "total": len(self.annotations),
            "positive": pos_count,
            "negative": neg_count,
            "pos_ratio": pos_count / len(self.annotations) if self.annotations else 0,
        }
