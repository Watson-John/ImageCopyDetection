#!/bin/bash
#SBATCH --job-name=ced_baseline_gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --partition=day-long-gpu
#SBATCH --nodelist=gpu4
#SBATCH --gres=gpu:A30:2
#SBATCH --output="/home/jowatson/Deep Learning/CEDetector_APT/logs/ced_baseline_gpu4_%j.out"
#SBATCH --error="/home/jowatson/Deep Learning/CEDetector_APT/logs/ced_baseline_gpu4_%j.err"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowatson@clarku.edu

set -euo pipefail

# Print job information
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Model: Baseline CED (no APT)"
echo "========================================="

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p "logs/job_${SLURM_JOB_ID}"

# Archive completed job logs to keep the directory tidy
if ! python3 "$(pwd)/logs/archive_logs.py" --logs-dir "$(pwd)/logs"; then
    echo "WARNING: log archiving failed; continuing" >&2
fi

# Load conda environment
echo "Activating conda environment: gpu_training"
if [ -f /opt/software/anaconda/3.9/etc/profile.d/conda.sh ]; then
    source /opt/software/anaconda/3.9/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Available CONDA_EXE=$CONDA_EXE" >&2
    exit 1
fi
conda activate gpu_training

# Verify GPU availability
echo "GPUs available:"
nvidia-smi --list-gpus

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501  # Different port from APT training

# NCCL optimizations for A30 GPUs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# Threading / dataloader tuning (per rank)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-20} - 2 ))
if [ "$NUM_WORKERS" -lt 2 ]; then NUM_WORKERS=2; fi

# PyTorch optimizations - Memory management for OOM prevention
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0  # Async kernel launches for speed

# Number of GPUs (A30 GPUs on gpu4)
NGPUS=2

# Training parameters (SAME AS APT for comparison)
EPOCHS=18  # Reduced from 25 to fit in 24h window
NDEC_EPOCHS=3  # Reduced from 5 to fit in 24h window
BATCH_SIZE=12  # Same as APT
LEARNING_RATE=2e-4
SUBSET_FRACTION=0.5  # Same 50% dataset as APT
USE_AMP=true  # Enable mixed precision training (FP16) - saves ~50% memory

# Paths
DISC21_PATH="/home/jowatson/Deep Learning/DISC21"
NDEC_PATH="/home/jowatson/Deep Learning/NDEC"
HF_TOKEN_PATH="/home/jowatson/Deep Learning/Code/.env"
SAVE_DIR="./checkpoints_baseline_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "Training Configuration:"
echo "  Model: Baseline CED (no APT)"
echo "  Epochs (DISC21): $EPOCHS"
echo "  Epochs (NDEC): $NDEC_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Subset Fraction: $SUBSET_FRACTION"
echo "  Mixed Precision: $USE_AMP"
echo "  Save Directory: $SAVE_DIR"
echo "  Number of GPUs: $NGPUS"
echo "  Num Workers (per rank): $NUM_WORKERS"
echo "========================================="

# Run training with DDP
echo "Starting baseline CED training (no APT)..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    train_baseline.py \
    --disc21_path "$DISC21_PATH" \
    --ndec_path "$NDEC_PATH" \
    --hf_token_path "$HF_TOKEN_PATH" \
    --epochs $EPOCHS \
    --ndec_epochs $NDEC_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --subset_fraction $SUBSET_FRACTION \
    --save_dir "$SAVE_DIR" \
    --save_every 5 \
    --dinov3_model "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --num_workers $NUM_WORKERS \
    --use_amp  # Enable mixed precision training

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "Checkpoints saved to: $SAVE_DIR"
    echo "========================================="
else
    echo "========================================="
    echo "Training failed with exit code $?"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi

# Print final checkpoint info
echo "Final model files:"
ls -lh "$SAVE_DIR"/*.pt

# Deactivate conda
conda deactivate
