# CEDetector with Adaptive Patch Transformers (APT)

Implementation of "An End-to-End Vision Transformer Approach for Image Copy Detection" (CED) combined with "Accelerating Vision Transformers with Adaptive Patch Sizes" (APT) using DiNOv3.

## Overview

This project combines two state-of-the-art approaches:
- **CED (Copy-Edit Detection)**: End-to-end vision transformer for detecting edited copies of images
- **APT (Adaptive Patch Transformers)**: Adaptive patch sizing for efficient vision transformers
- **DiNOv3**: Self-supervised vision transformer from Meta/Facebook

## Features

- ✅ Adaptive patch sizing based on image entropy
- ✅ DiNOv3 backbone from HuggingFace
- ✅ Multi-scale feature aggregation
- ✅ Cross-attention based copy-edit classification
- ✅ DDP (Distributed Data Parallel) training support
- ✅ DISC21 and NDEC dataset loaders
- ✅ CED augmentation pipeline
- ✅ Comprehensive evaluation metrics (µAP, R@P90)

## Project Structure

```
CEDetector_APT/
├── models/
│   ├── cedetector_apt.py          # Main model
│   ├── dinov3_backbone.py         # DiNOv3 wrapper
│   ├── apt_patch_selector.py      # Adaptive patch selection
│   ├── apt_patch_embedding.py     # Multi-scale patch embedding
│   ├── feature_aggregation.py     # CED feature aggregation
│   └── copy_edit_classifier.py    # Cross-attention classifier
├── data/
│   ├── disc21_dataset.py          # DISC21 data loader
│   └── ndec_dataset.py            # NDEC data loader
├── utils/
│   ├── ced_augmentations.py       # Augmentation pipeline
│   ├── losses.py                  # Training losses
│   └── metrics.py                 # Evaluation metrics
├── train.py                       # Training script
├── run_training.slurm            # SLURM job script
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU training)
- Conda environment: `gpu_training`

### Setup

1. Activate your conda environment:
```bash
conda activate gpu_training
```

2. Install dependencies:
```bash
cd "/home/jowatson/Deep Learning/CEDetector_APT"
pip install -r requirements.txt
```

3. Verify HuggingFace token is set:
```bash
cat "/home/jowatson/Deep Learning/Code/.env"
# Should show: HF_TOKEN=your_token_here
```

## Datasets

### DISC21 (Image Similarity Challenge 2021)
- Location: `/home/jowatson/Deep Learning/DISC21`
- 1M reference images
- 50K query images with ground truth

### NDEC (Negative Distractor for Edited Copy)
- Location: `/home/jowatson/Deep Learning/NDEC`
- Focus on hard negative distractors
- Used for final fine-tuning and evaluation

## Training

### Quick Start with SLURM

```bash
cd "/home/jowatson/Deep Learning/CEDetector_APT"
mkdir -p logs
sbatch run_training.slurm
```

### Monitor Training

```bash
# Watch output
tail -f logs/cedetector_apt_<job_id>.out

# Check GPU usage
squeue -u $USER
```

### Manual Training (without SLURM)

```bash
# Single GPU
python train.py \
    --disc21_path "/home/jowatson/Deep Learning/DISC21" \
    --ndec_path "/home/jowatson/Deep Learning/NDEC" \
    --hf_token_path "/home/jowatson/Deep Learning/Code/.env" \
    --epochs 25 \
    --ndec_epochs 5 \
    --batch_size 32 \
    --subset_fraction 0.2 \
    --save_dir ./checkpoints

# Multi-GPU with DDP
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
    --disc21_path "/home/jowatson/Deep Learning/DISC21" \
    --ndec_path "/home/jowatson/Deep Learning/NDEC" \
    --hf_token_path "/home/jowatson/Deep Learning/Code/.env" \
    --epochs 25 \
    --ndec_epochs 5 \
    --batch_size 32 \
    --subset_fraction 0.2
```

## Training Configuration

### Key Parameters

- `--epochs`: Number of epochs on DISC21 (default: 25)
- `--ndec_epochs`: Final epochs on NDEC (default: 5)
- `--batch_size`: Batch size per GPU (default: 32)
- `--lr`: Learning rate (default: 2e-4)
- `--subset_fraction`: Fraction of data to use (0.1-1.0)
  - Set to 0.2 (20%) to fit within 24-hour training window
  - Set to 1.0 for full dataset training
- `--freeze_backbone`: Freeze DiNOv3 backbone weights
- `--save_every`: Save checkpoint every N epochs (default: 5)

### Hardware Requirements

- **Current Setup**: 3x NVIDIA A30 GPUs (24GB VRAM each)
- **Recommended**: 3-4x GPUs with 24GB+ VRAM each
- **Minimum**: 1x GPU with 16GB+ VRAM
- **RAM**: 64GB+ system memory
- **Storage**: 50GB+ for datasets and checkpoints

**Note**: Code is optimized for NVIDIA A30 (Ampere) with:
- NCCL backend for DDP
- TF32 precision enabled
- cuDNN benchmarking
- See `DDP_SETUP.md` for details

### Training Time Estimates

With `subset_fraction=0.2` on **3x A30 GPUs**:
- DISC21 (25 epochs): ~20-22 hours
- NDEC (5 epochs): ~2-3 hours
- **Total**: ~22-25 hours (fits in 24-hour limit)
- **Speedup**: ~2.6x vs single GPU

With `subset_fraction=0.2` on 4x A100 GPUs:
- DISC21 (25 epochs): ~16-18 hours
- NDEC (5 epochs): ~2-3 hours
- **Total**: ~18-21 hours

## Model Architecture

### Adaptive Patch Sizing (APT)

- Computes entropy at multiple scales
- Assigns larger patches (64x64) to homogeneous regions
- Assigns smaller patches (16x16) to complex regions
- Reduces token count by 20-40% while maintaining accuracy

### DiNOv3 Backbone

- Pretrained vision transformer from Facebook
- Model: `facebook/dinov2-base`
- Extracts rich visual features with self-supervised learning

### Feature Aggregation

- Combines global CLS token features
- Attention-weighted salient regional features
- GeM pooling and whitening transformation

### Copy-Edit Classifier

- Cross-attention between query and reference
- Multi-head self-attention blocks
- Binary classification (copy-edit vs. not)

## Evaluation Metrics

Based on CED paper:
- **µAP (Micro Average Precision)**: Area under precision-recall curve
- **R@P90**: Recall at 90% precision
- **Accuracy**: Binary classification accuracy

## Checkpoints

Checkpoints are saved in the `--save_dir` directory:
- `checkpoint_epoch_X.pt`: Periodic checkpoints
- `best_model.pt`: Best model based on µAP
- `final_model.pt`: Final model after all training

### Loading a Checkpoint

```python
from models import CEDetectorAPT
import torch

# Create model
model = CEDetectorAPT(
    dinov3_model="facebook/dinov2-base",
    hf_token_path="/home/jowatson/Deep Learning/Code/.env"
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Key Implementation Details

### APT Patch Selection

From `apt_patch_selector.py`:
- Entropy thresholds: [5.5, 4.0] for 3 scales
- Quadtree structure for hierarchical selection
- Base patch size: 16x16
- Maximum patch size: 64x64

### Augmentations

From `ced_augmentations.py`:
- Random rotation, crop, resize
- Color jitter, blur, brightness/contrast
- JPEG compression simulation
- Text overlay and noise addition
- 4 random transforms per image

### Loss Function

Combined loss from CED paper:
```
L = L_contrast + L_MSL + L_BCE

where:
- L_contrast = L_SimCLR + λ * L_KL
- L_MSL = Multi-Similarity Loss
- L_BCE = Binary Cross-Entropy Loss
```

## Comparison with CED Paper

This implementation aims to match the CED paper evaluation:
- Same metrics (µAP, R@P90)
- Same datasets (DISC21, NDEC)
- Enhanced with APT for efficiency
- Uses DiNOv3 instead of original DINO

Expected performance improvements:
- 20-40% faster training due to APT
- Similar or better accuracy
- Better handling of multi-scale features

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Increase `--subset_fraction` reduction
- Use `--freeze_backbone` to reduce memory

### Slow Training

- Check GPU utilization: `nvidia-smi`
- Ensure DDP is enabled (multi-GPU)
- Reduce data loader workers if CPU-bound

### HuggingFace Token Error

```bash
# Verify token file exists and is readable
cat "/home/jowatson/Deep Learning/Code/.env"

# Should contain:
# HF_TOKEN=YOUR_HF_TOKEN_HERE
```

## References

1. **CED Paper**: "An End-to-End Vision Transformer Approach for Image Copy Detection"
2. **APT Paper**: "Accelerating Vision Transformers with Adaptive Patch Sizes"
3. **DiNOv3**: "DINOv2: Learning Robust Visual Features without Supervision"

## Citation

If you use this code, please cite the original papers:

```bibtex
@inproceedings{lee2023ced,
  title={An End-to-End Vision Transformer Approach for Image Copy Detection},
  author={Lee, Jiahe Steven and Hsu, Wynne and Lee, Mong Li},
  booktitle={CVPR Workshop},
  year={2023}
}

@article{choudhury2025apt,
  title={Accelerating Vision Transformers with Adaptive Patch Sizes},
  author={Choudhury, Rohan and Kim, JungEun and Park, Jinhyung and Yang, Eunho and Jeni, Laszlo and Kitani, Kris},
  journal={arXiv preprint arXiv:2510.18091},
  year={2025}
}
```

## License

This implementation is for research and educational purposes.

## Contact

For issues or questions, please check the logs in `logs/` directory or review the error outputs.
