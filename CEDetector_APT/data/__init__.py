"""Data loaders for CEDetector"""

from .disc21_dataset import DISC21Dataset, create_disc21_loaders
from .ndec_dataset import NDECDataset, create_ndec_loaders

__all__ = [
    'DISC21Dataset',
    'create_disc21_loaders',
    'NDECDataset',
    'create_ndec_loaders',
]
