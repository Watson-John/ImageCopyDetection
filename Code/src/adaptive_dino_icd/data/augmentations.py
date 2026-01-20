"""
Augmentation Pipeline for copy detection training.

Provides Albumentations-based transforms with optional AugLy integration.
Supports 50% application probability for hard augmentations.
"""

import logging
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Check AugLy availability
try:
    import augly.image as augly_image
    AUGLY_AVAILABLE = True
except ImportError:
    AUGLY_AVAILABLE = False
    logger.info("AugLy not available, using Albumentations only")

# Check Albumentations availability
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("Albumentations not available, using basic transforms")


class AugmentationPipeline:
    """
    Augmentation pipeline with configurable hard augmentation probability.

    Combines:
    - Basic augmentations (always applied): resize, normalize
    - Hard augmentations (applied with probability): geometric, photometric, etc.
    - Optional AugLy augmentations for realistic copy-edit transformations

    Args:
        image_size: Target image size (height, width) or single int for square
        hard_aug_prob: Probability of applying hard augmentations (default: 0.5)
        use_augly: Whether to use AugLy augmentations (if available)
        normalize_mean: Mean for normalization (default: ImageNet)
        normalize_std: Std for normalization (default: ImageNet)

    Example:
        pipeline = AugmentationPipeline(image_size=224, hard_aug_prob=0.5)
        transformed = pipeline(image)  # PIL Image or np.ndarray -> torch.Tensor
    """

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        hard_aug_prob: float = 0.5,
        use_augly: bool = False,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        self.hard_aug_prob = hard_aug_prob
        self.use_augly = use_augly and AUGLY_AVAILABLE
        self.normalize_mean = normalize_mean or self.IMAGENET_MEAN
        self.normalize_std = normalize_std or self.IMAGENET_STD

        # Build transforms
        self._build_transforms()

        if self.use_augly:
            logger.info("AugmentationPipeline: Using AugLy augmentations")
        else:
            logger.info("AugmentationPipeline: Using Albumentations only")

    def _build_transforms(self) -> None:
        """Build augmentation transforms."""
        if ALBUMENTATIONS_AVAILABLE:
            self._build_albumentations()
        else:
            self._build_basic_transforms()

    def _build_albumentations(self) -> None:
        """Build Albumentations-based transforms."""
        # Basic transforms (always applied)
        self.basic_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ])

        # Hard augmentations (applied with probability)
        self.hard_transform = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=1.0),
                A.Rotate(limit=30, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.ImageCompression(quality_range=(50, 90), p=1.0),
                A.Downscale(scale_range=(0.5, 0.9), p=1.0),
            ], p=0.3),
        ])

        # Combined hard + basic
        self.full_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            # Hard augmentations
            A.OneOf([
                A.RandomRotate90(p=1.0),
                A.Rotate(limit=30, border_mode=0, p=1.0),
            ], p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
            ], p=0.2),
            # Final normalize and convert
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ])

    def _build_basic_transforms(self) -> None:
        """Build basic transforms without Albumentations."""
        self.basic_transform = None
        self.hard_transform = None
        self.full_transform = None

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        apply_hard: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Apply augmentation pipeline to image.

        Args:
            image: Input image (PIL Image or numpy array)
            apply_hard: If None, randomly decide based on hard_aug_prob.
                       If True/False, force hard augmentation on/off.

        Returns:
            Augmented image as torch.Tensor (C, H, W)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Decide whether to apply hard augmentations
        if apply_hard is None:
            apply_hard = random.random() < self.hard_aug_prob

        # Apply AugLy first if enabled and hard augmentation requested
        if self.use_augly and apply_hard:
            image = self._apply_augly(image)

        # Apply Albumentations
        if ALBUMENTATIONS_AVAILABLE:
            if apply_hard:
                result = self.full_transform(image=image)
            else:
                result = self.basic_transform(image=image)
            return result["image"]
        else:
            return self._basic_transform_fallback(image)

    def _apply_augly(self, image: np.ndarray) -> np.ndarray:
        """Apply AugLy augmentations for realistic copy-edit transforms."""
        if not AUGLY_AVAILABLE:
            return image

        # Convert to PIL for AugLy
        pil_image = Image.fromarray(image)

        # Randomly select and apply AugLy transforms
        augly_transforms = [
            lambda img: augly_image.overlay_emoji(img, opacity=0.5, emoji_size=0.1),
            lambda img: augly_image.overlay_text(img, text=["COPY"], opacity=0.3),
            lambda img: augly_image.change_aspect_ratio(img, ratio=random.uniform(0.8, 1.2)),
            lambda img: augly_image.pad_square(img),
            lambda img: augly_image.perspective_transform(img, sigma=random.uniform(10, 30)),
        ]

        # Apply 0-2 random transforms
        n_transforms = random.randint(0, 2)
        if n_transforms > 0:
            selected = random.sample(augly_transforms, min(n_transforms, len(augly_transforms)))
            for transform in selected:
                try:
                    pil_image = transform(pil_image)
                except Exception as e:
                    logger.debug(f"AugLy transform failed: {e}")

        return np.array(pil_image)

    def _basic_transform_fallback(self, image: np.ndarray) -> torch.Tensor:
        """Fallback transform without Albumentations."""
        from PIL import Image
        import torchvision.transforms.functional as TF

        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Resize
        pil_image = pil_image.resize(self.image_size, Image.BILINEAR)

        # To tensor
        tensor = TF.to_tensor(pil_image)

        # Normalize
        tensor = TF.normalize(tensor, self.normalize_mean, self.normalize_std)

        return tensor


def get_train_transforms(
    image_size: int = 224,
    hard_aug_prob: float = 0.5,
    use_augly: bool = False,
) -> AugmentationPipeline:
    """
    Get training augmentation pipeline.

    Args:
        image_size: Target image size
        hard_aug_prob: Probability of hard augmentations
        use_augly: Whether to use AugLy

    Returns:
        AugmentationPipeline instance
    """
    return AugmentationPipeline(
        image_size=image_size,
        hard_aug_prob=hard_aug_prob,
        use_augly=use_augly,
    )


def get_val_transforms(
    image_size: int = 224,
) -> AugmentationPipeline:
    """
    Get validation/inference augmentation pipeline (no hard augmentations).

    Args:
        image_size: Target image size

    Returns:
        AugmentationPipeline instance with hard_aug_prob=0
    """
    return AugmentationPipeline(
        image_size=image_size,
        hard_aug_prob=0.0,
        use_augly=False,
    )


# Convenience function for direct use
def create_transform(
    image_size: int = 224,
    is_training: bool = True,
    hard_aug_prob: float = 0.5,
) -> Callable:
    """
    Create a transform function for datasets.

    Args:
        image_size: Target image size
        is_training: If True, use training augmentations
        hard_aug_prob: Probability of hard augmentations (training only)

    Returns:
        Callable transform
    """
    if is_training:
        return get_train_transforms(image_size, hard_aug_prob)
    else:
        return get_val_transforms(image_size)
