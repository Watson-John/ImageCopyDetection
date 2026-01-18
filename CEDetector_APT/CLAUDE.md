# CLAUDE.md - CEDetector with Adaptive Patch Transformers

This guide provides essential information for Claude Code instances working with the CEDetector_APT codebase.

## Project Overview

**CEDetector with APT** is a copy-edit detection system that combines:
- **CED (Copy-Edit Detection)**: End-to-end vision transformer for detecting edited copies of images
- **APT (Adaptive Patch Transformers)**: Adaptive patch sizing for efficient vision transformers
- **DiNOv3**: Self-supervised vision transformer backbone from Meta/Facebook via HuggingFace

The system trains on DISC21 dataset (1M reference images, 50K queries) and fine-tunes on NDEC dataset (hard negative distractors).

## Common Commands

### Environment Setup
```bash
# Activate conda environment
conda activate gpu_training

# Navigate to project directory
cd "/home/jowatson/Deep Learning/CEDetector_APT"

# Verify setup (checks packages, CUDA, datasets, HF token, model files)
python verify_setup.py
```

### Training

#### SLURM (Recommended for HPC)
```bash
# Create logs directory
mkdir -p logs

# Submit training job (3x A30 GPUs, 24-hour window)
sbatch run_training.slurm

# Monitor progress
tail -f logs/cedetector_apt_*.out

# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>
```

#### Interactive Training
```bash
# Single GPU (testing)
python train.py \
    --epochs 2 \
    --ndec_epochs 1 \
    --batch_size 16 \
    --subset_fraction 0.01 \
    --save_dir ./test_checkpoints

# Multi-GPU with DDP (3 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py \
    --disc21_path "/home/jowatson/Deep Learning/DISC21" \
    --ndec_path "/home/jowatson/Deep Learning/NDEC" \
    --hf_token_path "/home/jowatson/Deep Learning/Code/.env" \
    --epochs 25 \
    --ndec_epochs 5 \
    --batch_size 32 \
    --subset_fraction 0.2 \
    --save_dir ./checkpoints
```

### Monitoring
```bash
# Watch GPU usage (on compute node)
watch -n 1 nvidia-smi

# View training logs
tail -f logs/cedetector_apt_*.out

# Check error logs
tail -f logs/cedetector_apt_*.err

# Check job info
scontrol show job <job_id>
```

### Checkpoints
```bash
# List checkpoints
ls -lh checkpoints_*/

# Load best model in Python
python
>>> import torch
>>> from models import CEDetectorAPT
>>> model = CEDetectorAPT(
...     dinov3_model="facebook/dinov2-base",
...     hf_token_path="/home/jowatson/Deep Learning/Code/.env"
... )
>>> checkpoint = torch.load('checkpoints_*/best_model.pt')
>>> model.load_state_dict(checkpoint['model_state_dict'])
>>> print(f"Best µAP: {checkpoint['mu_ap']:.4f}")
```

## High-Level Architecture

### 1. Pipeline Overview

```
Query Image → APT Patch Selection → DiNOv3 Backbone → Feature Aggregation → Retrieval
                                                                          ↓
Reference Image → APT Patch Selection → DiNOv3 Backbone → Feature Aggregation
                                                                          ↓
                                              Query Tokens + Reference Tokens
                                                          ↓
                                              Copy-Edit Classifier → Similarity Score
```

### 2. Five Core Components

#### Component 1: APT Patch Selection (`models/apt_patch_selector.py`)
- **Purpose**: Adaptive patch sizing based on image entropy
- **Key Class**: `APTPatchSelector`
- **How it works**:
  - Computes entropy at multiple scales (16x16, 32x32, 64x64)
  - High entropy (complex regions) → smaller patches (16x16)
  - Low entropy (homogeneous regions) → larger patches (64x64)
  - Reduces token count by 20-40% while maintaining accuracy
- **Key Methods**:
  - `get_adaptive_patches()`: Extracts patches hierarchically using entropy thresholds
  - `extract_query_patches()`: Extracts fixed query patches (6 patches for CED)

#### Component 2: APT Patch Embedding (`models/apt_patch_embedding.py`)
- **Purpose**: Embed patches of different sizes into uniform token space
- **Key Class**: `APTPatchEmbedding` with `ZeroMLP`
- **How it works**:
  - Multi-scale embedding: Resize patch + aggregate sub-patches
  - Zero-initialized MLP for gradual integration (ControlNet-inspired)
  - Combines two pathways: direct resize + sub-patch aggregation
- **Innovation**: Allows applying APT to pretrained models without catastrophic forgetting

#### Component 3: DiNOv3 Backbone (`models/dinov3_backbone.py`)
- **Purpose**: Feature extraction from HuggingFace DiNOv3
- **Key Class**: `DiNOv3Backbone`
- **How it works**:
  - Wraps `facebook/dinov2-base` model
  - Loads HuggingFace token from `/home/jowatson/Deep Learning/Code/.env`
  - Returns CLS token, patch tokens, and attention weights
- **Optional**: `freeze_backbone` to reduce memory during fine-tuning

#### Component 4: CED Feature Aggregation (`models/feature_aggregation.py`)
- **Purpose**: Aggregate patch tokens into deep descriptors
- **Key Classes**: `CEDFeatureAggregation`, `GeMPooling`
- **How it works**:
  - Global features: CLS token from layer L
  - Salient features: Attention-weighted patch tokens (u = α_CLS ⊗ h_L)
  - Concatenate: descriptor = [z; u]
  - GeM pooling + whitening transformation
- **Output**: Deep descriptor for retrieval

#### Component 5: Copy-Edit Classifier (`models/copy_edit_classifier.py`)
- **Purpose**: Binary classification (copy-edit vs. not)
- **Key Class**: `CopyEditClassifier`
- **How it works**:
  - Cross-attention between query and reference tokens
  - Multi-head self-attention blocks (4 layers)
  - Global average pooling + binary classification
- **Formula**: Cross-attention = (hq·W1)·(hr·W2)^T / sqrt(d) · (hr·W3)

### 3. Main Model (`models/cedetector_apt.py`)

**Class**: `CEDetectorAPT`

**Orchestrates all components**:
1. `process_image_with_apt()`: Processes query/reference images
   - Query: Extracts 6 patches → APT → DiNOv3 → Features
   - Reference: Full image → DiNOv3 → Features
2. `build_reference_corpus()`: Indexes reference images for retrieval
3. `retrieve_candidates()`: k-NN retrieval using cosine similarity
4. `forward()`: End-to-end prediction
   - Extract query patches
   - Compare with references
   - Return similarity scores

### 4. Training Pipeline (`train.py`)

**DDP Setup**:
- NCCL backend for NVIDIA GPUs
- TF32 precision for A30 (Ampere) - ~3x speedup
- cuDNN benchmarking for optimal kernel selection
- Gradient bucketing for memory efficiency

**Training Phases**:
1. **DISC21 (25 epochs)**: Large-scale training on 1M reference images
2. **NDEC (5 epochs)**: Fine-tuning on hard negative distractors

**Key Functions**:
- `setup_ddp()`: Initialize distributed training
- `train_epoch()`: Single epoch training loop
- `validate()`: Validation with metrics

### 5. Data Loading (`data/`)

**DISC21Dataset** (`disc21_dataset.py`):
- Ground truth CSV: `dev_queries_groundtruth.csv`
- Reference images: 20 subdirectories (refs_50k_0 to refs_50k_19)
- Creates positive/negative pairs (50/50 split)

**NDECDataset** (`ndec_dataset.py`):
- Ground truth CSV: `public_ground_truth_h5.csv`
- Nested query structure: `queries_h5/{id}/query.jpg`
- Hard negative distractors (empty reference_id = negative sample)

**Augmentations** (`utils/ced_augmentations.py`):
- Random rotation, crop, resize
- Color jitter, blur, brightness/contrast
- JPEG compression simulation
- Text overlay and noise addition
- 4 random transforms per image

### 6. Loss Functions (`utils/losses.py`)

**Combined CED Loss**:
```
L = L_contrast + L_MSL + L_BCE

where:
- L_contrast = L_SimCLR + λ * L_KL (contrastive learning + regularization)
- L_MSL = Multi-Similarity Loss (metric learning)
- L_BCE = Binary Cross-Entropy (classification)
```

**Key Classes**:
- `SimCLRLoss`: NT-Xent contrastive loss
- `KLDivergenceLoss`: Regularizes attention distributions
- `MultiSimilarityLoss`: Metric learning loss
- `CEDLoss`: Combines all losses

### 7. Evaluation Metrics (`utils/metrics.py`)

Based on CED paper benchmarks:
- **µAP**: Micro Average Precision (area under precision-recall curve)
- **R@P90**: Recall at 90% precision
- **Accuracy**: Binary classification accuracy

**Class**: `MetricsTracker` - tracks and computes metrics during training

## Code Structure

```
CEDetector_APT/
├── models/                          # Model components
│   ├── cedetector_apt.py           # Main model (orchestrates all components)
│   ├── dinov3_backbone.py          # DiNOv3 wrapper
│   ├── apt_patch_selector.py       # Adaptive patch selection
│   ├── apt_patch_embedding.py      # Multi-scale patch embedding
│   ├── feature_aggregation.py      # CED feature aggregation
│   └── copy_edit_classifier.py     # Cross-attention classifier
├── data/                            # Data loaders
│   ├── disc21_dataset.py           # DISC21 loader
│   └── ndec_dataset.py             # NDEC loader
├── utils/                           # Utilities
│   ├── ced_augmentations.py        # Augmentation pipeline
│   ├── losses.py                   # Training losses
│   └── metrics.py                  # Evaluation metrics
├── train.py                        # DDP training script
├── run_training.slurm              # SLURM job script (3x A30 GPUs)
├── verify_setup.py                 # Setup verification
├── README.md                       # Full documentation
├── DDP_SETUP.md                    # DDP configuration guide
└── QUICKSTART.md                   # Quick start guide
```

## Key Configuration

### Hardware
- **Current Setup**: 3x NVIDIA A30 GPUs (24GB VRAM each)
- **Conda Environment**: `gpu_training`
- **Backend**: NCCL for distributed training

### Paths
- **DISC21 Dataset**: `/home/jowatson/Deep Learning/DISC21`
- **NDEC Dataset**: `/home/jowatson/Deep Learning/NDEC`
- **HuggingFace Token**: `/home/jowatson/Deep Learning/Code/.env`
- **Checkpoints**: `./checkpoints_<timestamp>/`

### Training Parameters (run_training.slurm)
- **DISC21 Epochs**: 25
- **NDEC Epochs**: 5
- **Batch Size**: 32 per GPU (global batch = 96 with 3 GPUs)
- **Learning Rate**: 2e-4
- **Subset Fraction**: 0.2 (20% of data for 24-hour window)
- **Save Every**: 5 epochs

### APT Parameters
- **Base Patch Size**: 16x16
- **Num Scales**: 3 (16x16, 32x32, 64x64)
- **Entropy Thresholds**: [5.5, 4.0]
- **Num Query Patches**: 6

## Important Implementation Details

### 1. Zero-Initialized MLP
From APT paper - gradual integration approach:
```python
# In apt_patch_embedding.py
class ZeroMLP(nn.Module):
    def __init__(self, dim: int):
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)  # Zero initialization
        nn.init.zeros_(self.linear.bias)
```

### 2. Query Patch Extraction
CED uses 6 fixed patches per query image:
- Ensures comprehensive coverage
- Each patch processed independently
- Maximum score across patches used for final prediction

### 3. DDP Optimizations
```python
# In train.py
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for Ampere
torch.backends.cudnn.benchmark = True         # Auto-tuning
model = DDP(model, gradient_as_bucket_view=True)  # Memory efficiency
```

### 4. Import Alias
`CEDAugmentations` is an alias for `CEDAugmentationPipeline`:
```python
# In utils/__init__.py
from .ced_augmentations import CEDAugmentationPipeline
CEDAugmentations = CEDAugmentationPipeline
```

## Performance Expectations

### Training Time (20% subset, 3x A30 GPUs)
- **DISC21 (25 epochs)**: ~20-22 hours
- **NDEC (5 epochs)**: ~2-3 hours
- **Total**: ~22-25 hours (fits in 24-hour limit)
- **Speedup**: ~2.6x vs single GPU

### Expected Metrics
**DISC21 Dataset**:
- µAP: ~0.85-0.87
- R@P90: ~0.80-0.85
- Accuracy: ~0.90-0.92

**NDEC Dataset** (harder):
- µAP: ~0.65-0.70
- R@P90: ~0.65-0.70
- Accuracy: ~0.80-0.85

### APT Benefits
- 20-40% faster training (reduced token count)
- Similar or better accuracy
- Better multi-scale feature extraction

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in run_training.slurm
BATCH_SIZE=24  # instead of 32

# Or reduce subset
SUBSET_FRACTION=0.15  # instead of 0.2
```

### Slow Training
```bash
# Check GPU utilization (should be 90-100%)
nvidia-smi

# Ensure DDP is working
grep "DDP initialized" logs/cedetector_apt_*.out
```

### HuggingFace Token Error
```bash
# Verify token exists
cat "/home/jowatson/Deep Learning/Code/.env"
# Should contain: HF_TOKEN=YOUR_HF_TOKEN_HERE
```

### NCCL Errors
```bash
# Check NCCL logs
grep "NCCL" logs/cedetector_apt_*.out

# Should see:
# NCCL INFO Ring 00 : 0 1 2
# NCCL INFO Channel 00 : 0 -> 1 -> 2 -> 0
```

## References

1. **CED Paper**: "An End-to-End Vision Transformer Approach for Image Copy Detection"
   - Path: `/home/jowatson/Deep Learning/CED.pdf`
2. **APT Paper**: "Accelerating Vision Transformers with Adaptive Patch Sizes"
   - Path: `/home/jowatson/Deep Learning/APT.pdf`
3. **DiNOv3**: "DINOv2: Learning Robust Visual Features without Supervision"
   - Model: `facebook/dinov2-base` (HuggingFace)

## Quick Reference

**Most Important Files**:
- `models/cedetector_apt.py`: Main model
- `train.py`: Training script
- `run_training.slurm`: SLURM job configuration
- `verify_setup.py`: Setup verification

**Most Common Tasks**:
1. Verify setup: `python verify_setup.py`
2. Submit training: `sbatch run_training.slurm`
3. Monitor training: `tail -f logs/cedetector_apt_*.out`
4. Check GPU usage: `nvidia-smi` (on compute node)

**Key Concepts to Remember**:
- APT uses entropy thresholds [5.5, 4.0] for 3 scales
- CED extracts 6 query patches per image
- Training uses DISC21 (25 epochs) then NDEC (5 epochs)
- DDP uses NCCL backend with TF32 precision for A30 GPUs
- Subset fraction 0.2 fits training in 24-hour window
