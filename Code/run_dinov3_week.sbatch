#!/bin/bash
#SBATCH --job-name=dinov3_ced_week
#SBATCH --partition=week-long-gpu
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:A30:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=512000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jowatson@clarku.edu
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-dinov3-week-%j.out
#SBATCH --error=slurm-dinov3-week-%j.err
#SBATCH --chdir="/home/jowatson/Deep Learning/Code"

set -euo pipefail

mkdir -p logs

{
    echo "Starting job at $(date)"
    echo "Running on node: ${SLURMD_NODENAME:-unknown}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
} >> logs/dinov3-week-status.log


GPU_ENV_PY="/home/jowatson/.conda/envs/gpu_training/bin/python"
if [ ! -x "$GPU_ENV_PY" ]; then
    echo "[ERROR] Expected python interpreter at $GPU_ENV_PY" >&2
    exit 1
fi

"$GPU_ENV_PY" -u DINOv3_CED_SLURM.py

echo "Job finished at $(date)" >> logs/dinov3-week-status.log
