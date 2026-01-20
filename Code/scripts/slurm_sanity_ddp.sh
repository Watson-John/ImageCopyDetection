#!/bin/bash
#SBATCH --job-name=sanity_ddp
#SBATCH --partition=short-gpu
#SBATCH --nodelist=gpu4
#SBATCH --gres=gpu:A30:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_ddp_%j.out
#SBATCH --error=logs/sanity_ddp_%j.err
#SBATCH --ntasks=1

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "DDP SANITY CHECK"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Load conda
source /opt/software/anaconda/3.9/etc/profile.d/conda.sh
conda activate gpu_training

# Set working directory
cd "/home/jowatson/Deep Learning/Code"

# Load environment variables
export $(cat .env | xargs)

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# Test DDP with 2 GPUs and short training (2 epochs, real model, small data subset)
echo "Testing DDP with 2 GPUs and real DINOv3 model..."
torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_phase1_ddp.py \
    --config configs/phase1_ddp_test.yaml

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
