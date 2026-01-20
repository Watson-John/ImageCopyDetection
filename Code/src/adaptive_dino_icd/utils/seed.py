"""
Reproducibility utilities for setting random seeds.

Ensures deterministic behavior across runs when properly seeded.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable PyTorch deterministic algorithms
                      (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass  # Some operations may not have deterministic implementations


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a seeded PyTorch generator for reproducible data loading.

    Args:
        seed: Random seed (if None, uses random initialization)

    Returns:
        Seeded torch.Generator instance

    Example:
        generator = get_generator(42)
        dataloader = DataLoader(dataset, generator=generator)
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader to ensure reproducibility.

    Use this as the worker_init_fn parameter in DataLoader.

    Args:
        worker_id: Worker process ID

    Example:
        dataloader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=worker_init_fn
        )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
