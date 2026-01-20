"""
DISC Dataset for self-supervised copy detection training.

Loads unlabeled images and generates (full, crop) pairs for contrastive learning.
The full image is the reference, the crop is the query.
"""

import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DISCDataset(Dataset):
    """
    DISC-style dataset for self-supervised image copy detection.

    Loads images from a folder and generates (full, crop) positive pairs.
    The full image is always the reference; the crop is the query.
    Direction is always "full->crop".

    Args:
        root_dir: Path to folder containing images
        transform: Transform to apply to both full and crop images
        crop_transform: Additional transform for crop generation (before main transform)
        crop_ratio_range: (min_ratio, max_ratio) for random crop size
        image_extensions: Valid image file extensions
        max_samples: Maximum number of samples (None for all)

    Yields per __getitem__:
        Dictionary with:
            - img_full: Full image tensor
            - img_crop: Cropped image tensor
            - stream: "disc"
            - direction: "full->crop"
            - is_copy: True (always positive pair)
            - path: Original image path
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        crop_transform: Optional[Callable] = None,
        crop_ratio_range: Tuple[float, float] = (0.3, 0.8),
        image_extensions: Optional[set] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.crop_transform = crop_transform
        self.crop_ratio_range = crop_ratio_range
        self.extensions = image_extensions or self.VALID_EXTENSIONS

        # Find all image files
        self.image_paths = self._find_images()

        if max_samples is not None and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]

        logger.info(f"DISCDataset: Found {len(self.image_paths)} images in {root_dir}")

    def _find_images(self) -> List[Path]:
        """Find all valid image files in root directory."""
        if not self.root_dir.exists():
            logger.warning(f"DISC root directory does not exist: {self.root_dir}")
            return []

        images = []
        for ext in self.extensions:
            images.extend(self.root_dir.glob(f"*{ext}"))
            images.extend(self.root_dir.glob(f"*{ext.upper()}"))
            # Also search subdirectories
            images.extend(self.root_dir.glob(f"**/*{ext}"))
            images.extend(self.root_dir.glob(f"**/*{ext.upper()}"))

        # Remove duplicates and sort for reproducibility
        images = sorted(list(set(images)))
        return images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """Get a (full, crop) positive pair."""
        img_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a dummy sample on error
            return self._get_dummy_sample(str(img_path))

        # Generate random crop
        crop_image = self._random_crop(image)

        # Apply crop-specific transform if provided
        if self.crop_transform is not None:
            crop_image = self.crop_transform(crop_image)

        # Apply main transforms
        if self.transform is not None:
            img_full = self.transform(image)
            img_crop = self.transform(crop_image)
        else:
            # Default: convert to tensor
            img_full = self._default_transform(image)
            img_crop = self._default_transform(crop_image)

        return {
            "img_full": img_full,
            "img_crop": img_crop,
            "stream": "disc",
            "direction": "full->crop",
            "is_copy": True,
            "path": str(img_path),
        }

    def _random_crop(self, image: Image.Image) -> Image.Image:
        """Generate a random crop from the image."""
        W, H = image.size

        # Random crop ratio
        ratio = random.uniform(*self.crop_ratio_range)

        # Crop dimensions
        crop_w = int(W * ratio)
        crop_h = int(H * ratio)

        # Random position
        max_x = W - crop_w
        max_y = H - crop_h

        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))

        # Crop
        crop = image.crop((x, y, x + crop_w, y + crop_h))

        return crop

    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Default transform: resize and convert to tensor."""
        import torchvision.transforms.functional as TF

        # Resize to standard size
        image = image.resize((224, 224), Image.BILINEAR)

        # Convert to tensor and normalize to [0, 1]
        tensor = TF.to_tensor(image)

        return tensor

    def _get_dummy_sample(self, path: str) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """Return dummy sample for error cases."""
        dummy = torch.zeros(3, 224, 224)
        return {
            "img_full": dummy,
            "img_crop": dummy,
            "stream": "disc",
            "direction": "full->crop",
            "is_copy": True,
            "path": path,
        }


class DISCDatasetWithHardCrops(DISCDataset):
    """
    Extended DISC dataset with harder crop augmentations.

    Generates both easy crops (simple random crop) and hard crops
    (with additional transformations like rotation, color jitter).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        crop_transform: Optional[Callable] = None,
        crop_ratio_range: Tuple[float, float] = (0.3, 0.8),
        hard_crop_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            crop_transform=crop_transform,
            crop_ratio_range=crop_ratio_range,
            **kwargs,
        )
        self.hard_crop_prob = hard_crop_prob

    def _random_crop(self, image: Image.Image) -> Image.Image:
        """Generate crop with optional hard augmentations."""
        crop = super()._random_crop(image)

        # Apply hard augmentations with probability
        if random.random() < self.hard_crop_prob:
            crop = self._apply_hard_augmentation(crop)

        return crop

    def _apply_hard_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply random hard augmentation to crop."""
        augmentations = [
            self._random_rotation,
            self._random_flip,
            self._random_color_jitter,
        ]

        # Apply 1-2 random augmentations
        n_augs = random.randint(1, 2)
        selected = random.sample(augmentations, n_augs)

        for aug in selected:
            image = aug(image)

        return image

    def _random_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation."""
        angle = random.uniform(-30, 30)
        return image.rotate(angle, expand=True, fillcolor=(128, 128, 128))

    def _random_flip(self, image: Image.Image) -> Image.Image:
        """Apply random horizontal flip."""
        if random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _random_color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply random color jitter."""
        from PIL import ImageEnhance

        # Random brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))

        # Random contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))

        return image
