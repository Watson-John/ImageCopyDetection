"""
Configuration management using dataclasses and YAML.

Provides typed configuration objects for all components of the Adaptive-DINO-ICD system.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml


@dataclass
class BackboneConfig:
    """Configuration for the DINOv3 backbone."""

    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    offline_stub: bool = False
    pretrained: bool = True
    embed_dim: int = 768
    freeze_backbone: bool = False

    def __post_init__(self):
        if self.offline_stub and self.model_name == "facebook/dinov3-vitb16-pretrain-lvd1689m":
            # Auto-switch to timm model for offline stub
            self.model_name = "vit_base_patch16_224"


@dataclass
class APTConfig:
    """Configuration for Adaptive Patch Tokenization."""

    entropy_scales: List[int] = field(default_factory=lambda: [8, 16, 32])
    min_patch_size: int = 8
    max_patch_size: int = 64
    entropy_thresholds: Dict[int, float] = field(
        default_factory=lambda: {8: 0.3, 16: 0.5, 32: 0.7}
    )
    normalize_entropy: bool = True
    embed_dim: int = 768
    smallest_patch_size: int = 16  # Base patch size for embedding


@dataclass
class LossConfig:
    """Configuration for ASL loss module."""

    lambda_mtr: float = 0.5  # Weight for metric loss term
    temperature: float = 0.07  # Temperature for contrastive loss
    margin: float = 0.2  # Margin for metric loss
    use_hard_negatives: bool = True
    normalize_features: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""

    disc_root: str = "./data/disc"
    ndec_annotation: str = "./data/ndec/pairs.csv"
    ndec_image_root: str = "./data/ndec/images"
    image_size: int = 224
    disc_ratio: float = 0.7  # 70% DISC, 30% NDEC
    batch_size: int = 32
    num_workers: int = 4
    crop_ratio_range: List[float] = field(default_factory=lambda: [0.3, 0.8])
    use_augly: bool = False
    hard_aug_prob: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    log_interval: int = 100  # Log every N steps
    save_interval: int = 1  # Save checkpoint every N epochs
    gradient_clip: float = 1.0
    warmup_steps: int = 0  # No warmup by default


@dataclass
class Config:
    """
    Master configuration combining all component configs.

    Example usage:
        config = load_config("configs/phase1_default.yaml")
        model = AdaptiveBackbone(config.backbone, config.apt)
    """

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    apt: APTConfig = field(default_factory=APTConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            backbone=BackboneConfig(**config_dict.get("backbone", {})),
            apt=APTConfig(**config_dict.get("apt", {})),
            loss=LossConfig(**config_dict.get("loss", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "backbone": asdict(self.backbone),
            "apt": asdict(self.apt),
            "loss": asdict(self.loss),
            "data": asdict(self.data),
            "training": asdict(self.training),
        }


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with loaded settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    return Config.from_dict(config_dict)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path to save YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
