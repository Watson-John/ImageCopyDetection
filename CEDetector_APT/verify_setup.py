#!/usr/bin/env python3
"""
Verification script to check if everything is set up correctly
before running training.
"""

import os
import sys
from pathlib import Path

def check_mark(success):
    return "✓" if success else "✗"

def verify_setup():
    """Verify all components are in place."""
    print("=" * 60)
    print("CEDetector with APT - Setup Verification")
    print("=" * 60)

    all_checks_passed = True

    # Check Python packages
    print("\n1. Checking Python packages...")
    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'PIL',
        'pandas',
        'numpy',
        'sklearn',
        'dotenv',
    ]

    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            elif package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"  {check_mark(True)} {package}")
        except ImportError:
            print(f"  {check_mark(False)} {package} - MISSING")
            all_checks_passed = False

    # Check CUDA
    print("\n2. Checking CUDA availability...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  {check_mark(cuda_available)} CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  {check_mark(True)} GPU count: {torch.cuda.device_count()}")
            print(f"  {check_mark(True)} GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ Warning: CUDA not available, training will be slow")
    except Exception as e:
        print(f"  {check_mark(False)} Error checking CUDA: {e}")
        all_checks_passed = False

    # Check datasets
    print("\n3. Checking datasets...")
    disc21_path = Path("/home/jowatson/Deep Learning/DISC21")
    ndec_path = Path("/home/jowatson/Deep Learning/NDEC")

    disc21_exists = disc21_path.exists()
    ndec_exists = ndec_path.exists()

    print(f"  {check_mark(disc21_exists)} DISC21 dataset: {disc21_path}")
    if disc21_exists:
        gt_file = disc21_path / "dev_queries_groundtruth.csv"
        print(f"    {check_mark(gt_file.exists())} Ground truth file: {gt_file.name}")
        query_dir = disc21_path / "dev_queries_50k_0"
        print(f"    {check_mark(query_dir.exists())} Query directory: {query_dir.name}")
    else:
        all_checks_passed = False

    print(f"  {check_mark(ndec_exists)} NDEC dataset: {ndec_path}")
    if ndec_exists:
        gt_file = ndec_path / "public_ground_truth_h5.csv"
        print(f"    {check_mark(gt_file.exists())} Ground truth file: {gt_file.name}")
    else:
        all_checks_passed = False

    # Check HuggingFace token
    print("\n4. Checking HuggingFace token...")
    token_path = Path("/home/jowatson/Deep Learning/Code/.env")
    token_exists = token_path.exists()
    print(f"  {check_mark(token_exists)} Token file: {token_path}")

    if token_exists:
        try:
            with open(token_path) as f:
                content = f.read()
                has_token = 'HF_TOKEN=' in content
                print(f"  {check_mark(has_token)} HF_TOKEN variable found")
                if not has_token:
                    all_checks_passed = False
        except Exception as e:
            print(f"  {check_mark(False)} Error reading token file: {e}")
            all_checks_passed = False
    else:
        all_checks_passed = False

    # Check model files
    print("\n5. Checking model files...")
    model_files = [
        'models/__init__.py',
        'models/cedetector_apt.py',
        'models/dinov3_backbone.py',
        'models/apt_patch_selector.py',
        'models/apt_patch_embedding.py',
        'models/feature_aggregation.py',
        'models/copy_edit_classifier.py',
    ]

    for file in model_files:
        file_path = Path(file)
        exists = file_path.exists()
        print(f"  {check_mark(exists)} {file}")
        if not exists:
            all_checks_passed = False

    # Check data loaders
    print("\n6. Checking data loaders...")
    data_files = [
        'data/__init__.py',
        'data/disc21_dataset.py',
        'data/ndec_dataset.py',
    ]

    for file in data_files:
        file_path = Path(file)
        exists = file_path.exists()
        print(f"  {check_mark(exists)} {file}")
        if not exists:
            all_checks_passed = False

    # Check utilities
    print("\n7. Checking utilities...")
    util_files = [
        'utils/__init__.py',
        'utils/ced_augmentations.py',
        'utils/losses.py',
        'utils/metrics.py',
    ]

    for file in util_files:
        file_path = Path(file)
        exists = file_path.exists()
        print(f"  {check_mark(exists)} {file}")
        if not exists:
            all_checks_passed = False

    # Check training script
    print("\n8. Checking training scripts...")
    train_exists = Path('train.py').exists()
    slurm_exists = Path('run_training.slurm').exists()

    print(f"  {check_mark(train_exists)} train.py")
    print(f"  {check_mark(slurm_exists)} run_training.slurm")

    if not (train_exists and slurm_exists):
        all_checks_passed = False

    # Test import
    print("\n9. Testing imports...")
    try:
        from models import CEDetectorAPT
        print(f"  {check_mark(True)} Can import CEDetectorAPT")
    except Exception as e:
        print(f"  {check_mark(False)} Error importing CEDetectorAPT: {e}")
        all_checks_passed = False

    try:
        from data import DISC21Dataset, NDECDataset
        print(f"  {check_mark(True)} Can import dataset loaders")
    except Exception as e:
        print(f"  {check_mark(False)} Error importing datasets: {e}")
        all_checks_passed = False

    try:
        from utils import CEDAugmentations, CEDLoss, MetricsTracker
        print(f"  {check_mark(True)} Can import utilities")
    except Exception as e:
        print(f"  {check_mark(False)} Error importing utilities: {e}")
        all_checks_passed = False

    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ All checks passed! You're ready to start training.")
        print("\nTo start training, run:")
        print("  sbatch run_training.slurm")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1
    print("=" * 60)

    return 0

if __name__ == '__main__':
    exit_code = verify_setup()
    sys.exit(exit_code)
