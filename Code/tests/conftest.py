"""
Pytest configuration and fixtures for Adaptive-DINO-ICD tests.
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cpu")  # Use CPU for tests to avoid GPU requirements


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    return seed_val


@pytest.fixture
def uniform_image():
    """Create a uniform (low entropy) image tensor."""
    # Uniform gray image
    return torch.full((1, 3, 224, 224), 0.5)


@pytest.fixture
def textured_image():
    """Create a textured (high entropy) image tensor."""
    # Random noise image
    torch.manual_seed(42)
    return torch.rand(1, 3, 224, 224)


@pytest.fixture
def batch_images():
    """Create a batch of test images."""
    torch.manual_seed(42)
    return torch.rand(4, 3, 224, 224)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from adaptive_dino_icd.utils.config import (
        Config, BackboneConfig, APTConfig, LossConfig, DataConfig, TrainingConfig
    )

    return Config(
        backbone=BackboneConfig(
            offline_stub=True,  # Use stub for testing
            embed_dim=768,
        ),
        apt=APTConfig(
            entropy_scales=[8, 16, 32],
            min_patch_size=8,
            max_patch_size=64,
        ),
        loss=LossConfig(
            lambda_mtr=0.5,
            temperature=0.07,
            margin=0.2,
        ),
        data=DataConfig(
            batch_size=4,
            disc_ratio=0.7,
        ),
        training=TrainingConfig(
            epochs=2,
            seed=42,
        ),
    )


@pytest.fixture
def entropy_scorer():
    """Create an EntropyScorer instance."""
    from adaptive_dino_icd.apt import EntropyScorer
    return EntropyScorer(scales=[8, 16, 32], normalize=True)


@pytest.fixture
def patch_selector():
    """Create a PatchSelector instance."""
    from adaptive_dino_icd.apt import PatchSelector
    return PatchSelector(
        min_patch_size=8,
        max_patch_size=64,
        entropy_thresholds={64: 0.3, 32: 0.5, 16: 0.7},
    )


@pytest.fixture
def patch_aggregator():
    """Create a PatchAggregator instance."""
    from adaptive_dino_icd.apt import PatchAggregator
    return PatchAggregator(embed_dim=768, smallest_patch_size=16)


@pytest.fixture
def asl_loss():
    """Create an ASLLossModule instance."""
    from adaptive_dino_icd.losses import ASLLossModule
    return ASLLossModule(lambda_mtr=0.5, temperature=0.07, margin=0.2)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary directories for test data."""
    disc_dir = tmp_path / "disc"
    disc_dir.mkdir()

    ndec_dir = tmp_path / "ndec"
    ndec_images = ndec_dir / "images"
    ndec_images.mkdir(parents=True)

    # Create some dummy images for DISC
    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(i * 25, i * 25, i * 25))
        img.save(disc_dir / f"img_{i:03d}.jpg")

    # Create some dummy images for NDEC
    for i in range(20):
        img = Image.new("RGB", (100, 100), color=(i * 12, i * 12, i * 12))
        img.save(ndec_images / f"pair_{i:03d}.jpg")

    # Create NDEC annotation file
    import json
    pairs = []
    for i in range(0, 20, 2):
        pairs.append({
            "img_a_path": f"pair_{i:03d}.jpg",
            "img_b_path": f"pair_{i+1:03d}.jpg",
            "label": "pos" if i % 4 == 0 else "neg",
            "direction": "a->b",
            "similar_pair_for_metric": i % 4 == 0,
        })

    with open(ndec_dir / "pairs.json", "w") as f:
        json.dump(pairs, f)

    return {
        "disc": disc_dir,
        "ndec": ndec_dir,
        "ndec_images": ndec_images,
        "ndec_annotation": ndec_dir / "pairs.json",
    }
