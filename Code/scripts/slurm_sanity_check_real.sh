#!/bin/bash
#SBATCH --job-name=sanity_real
#SBATCH --partition=short-gpu
#SBATCH --nodelist=gpu4
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_real_%j.out
#SBATCH --error=logs/sanity_real_%j.err

# Create logs directory if it doesn't exist
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
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run sanity check with real data and real model
echo "Running sanity check with REAL data and REAL DINOv3 model..."
python scripts/sanity_check_real_data.py --batch_size 4 --num_batches 5

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
