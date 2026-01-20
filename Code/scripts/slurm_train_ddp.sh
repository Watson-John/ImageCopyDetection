#!/bin/bash
#SBATCH --job-name=train_ddp
#SBATCH --partition=week-long-gpu
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:A30:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train_ddp_%j.out
#SBATCH --error=logs/train_ddp_%j.err
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jowatson@clarku.edu

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Load conda
source /opt/software/anaconda/3.9/etc/profile.d/conda.sh
conda activate gpu_training

# Set working directory
cd "/home/jowatson/Deep Learning/Code"

# Load environment variables (HuggingFace token)
export $(cat .env | xargs)

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=3

# Run DDP training with torchrun (spawns 3 processes for 3 GPUs)
echo "Starting DDP training with 3 GPUs..."
torchrun \
    --nproc_per_node=3 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_phase1_ddp.py \
    --config configs/phase1_ddp_test.yaml

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
