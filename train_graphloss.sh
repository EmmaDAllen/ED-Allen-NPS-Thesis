#!/bin/bash
#SBATCH --job-name=train_graphloss
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm/train_graphloss_%A_%a.out
#SBATCH --error=slurm/train_graphloss_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

MODELS=("tropical_v2" "edge_transformer" "gnn")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT_TAG="mixed_filtered_graphloss"

echo "========================================"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Experiment Tag: ${EXPERIMENT_TAG}"
echo "Training Seed: 1"
echo "========================================"

PYTHONPATH=. python -u experiments/train.py \
    "$MODEL" shortest_path "$EXPERIMENT_TAG"

echo "Completed training: ${MODEL} / ${EXPERIMENT_TAG}"

