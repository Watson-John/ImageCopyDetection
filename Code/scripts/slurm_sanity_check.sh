#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --partition=short-gpu
#SBATCH --nodelist=gpu4
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_check_%j.out
#SBATCH --error=logs/sanity_check_%j.err

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

# Load environment variables
export $(cat .env | xargs)

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run sanity check
echo "Running sanity check..."
python scripts/sanity_check_forward.py

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
