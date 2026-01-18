# Quick Start Guide - CEDetector with APT

## Verification

First, verify your setup is correct:

```bash
cd "/home/jowatson/Deep Learning/CEDetector_APT"
conda activate gpu_training
python verify_setup.py
```

You should see all checks pass with ✓ marks.

## Start Training

### Option 1: SLURM (Recommended for HPC)

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch run_training.slurm

# Monitor progress
tail -f logs/cedetector_apt_*.out

# Check job status
squeue -u $USER
```

### Option 2: Interactive (for testing)

```bash
# Activate environment
conda activate gpu_training

# Run with reduced settings for testing
python train.py \
    --epochs 2 \
    --ndec_epochs 1 \
    --batch_size 16 \
    --subset_fraction 0.01 \
    --save_dir ./test_checkpoints
```

## Training Configuration

The SLURM script is configured for 24-hour training with:
- **Epochs**: 25 (DISC21) + 5 (NDEC)
- **Batch size**: 32 per GPU
- **GPUs**: 4
- **Subset**: 20% of data
- **Estimated time**: ~22-23 hours

### Adjust for Different Time Limits

Edit `run_training.slurm`:

**For 12 hours:**
```bash
EPOCHS=12
NDEC_EPOCHS=2
SUBSET_FRACTION=0.15
```

**For full dataset (>48 hours):**
```bash
EPOCHS=30
NDEC_EPOCHS=5
SUBSET_FRACTION=1.0
```

## Monitor Training

### Check GPU Usage
```bash
# On the compute node
nvidia-smi

# Or via SLURM
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

### View Training Logs
```bash
# Real-time monitoring
tail -f logs/cedetector_apt_<job_id>.out

# Check for errors
tail -f logs/cedetector_apt_<job_id>.err
```

## After Training

### Checkpoints Location

Checkpoints are saved to `./checkpoints_<timestamp>/`:
- `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, etc.
- `best_model.pt` - Best model based on µAP
- `final_model.pt` - Final model after all training

### Evaluate Model

```python
import torch
from models import CEDetectorAPT

# Load model
model = CEDetectorAPT(
    dinov3_model="facebook/dinov2-base",
    hf_token_path="/home/jowatson/Deep Learning/Code/.env"
)

checkpoint = torch.load('checkpoints_*/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check metrics
print(f"Best µAP: {checkpoint['mu_ap']:.4f}")
```

## Troubleshooting

### Job Failed to Start
```bash
# Check SLURM errors
cat logs/cedetector_apt_<job_id>.err

# Verify GPU availability
sinfo -o "%20N  %10c  %10m  %25f  %10G"
```

### Out of Memory
```bash
# Reduce batch size in run_training.slurm
BATCH_SIZE=16  # instead of 32

# Or reduce subset
SUBSET_FRACTION=0.1  # instead of 0.2
```

### Slow Training
```bash
# Ensure using multiple GPUs
NGPUS=4  # in run_training.slurm

# Check dataloader workers
# In train.py, num_workers=4 is default
```

## Expected Results

Based on CED paper benchmarks:

### DISC21 Dataset
- **µAP**: ~0.85-0.87
- **R@P90**: ~0.80-0.85
- **Accuracy**: ~0.90-0.92

### NDEC Dataset (harder)
- **µAP**: ~0.65-0.70
- **R@P90**: ~0.65-0.70
- **Accuracy**: ~0.80-0.85

APT should provide:
- 20-40% faster training
- Similar or better accuracy
- Better multi-scale feature extraction

## Key Files

```
CEDetector_APT/
├── train.py              # Main training script
├── run_training.slurm    # SLURM job configuration
├── verify_setup.py       # Setup verification
├── README.md             # Full documentation
└── QUICKSTART.md         # This file
```

## Getting Help

1. Check verification output: `python verify_setup.py`
2. Review training logs: `tail -f logs/*.out`
3. Check error logs: `tail -f logs/*.err`
4. Verify GPU allocation: `nvidia-smi` (on compute node)

## Next Steps

After successful training:
1. Evaluate on test set
2. Compare with baseline (non-APT) model
3. Analyze performance on different image types
4. Test on custom copy-edit detection tasks
