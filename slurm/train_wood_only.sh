#!/bin/bash
#SBATCH --job-name=train_wood_only
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/train_wood_only_%A_%a.out
#SBATCH --error=logs/train_wood_only_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

MODELS=("tropical_v2" "edge_transformer" "gnn")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_TAG="wood_only"
DATASET="data/training_data_shortest_path_wood_only.json"

if [[ ! -f "$DATASET" ]]; then
    echo "Missing training dataset: $DATASET" >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID | Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL | Experiment: $EXPERIMENT_TAG"
echo "Dataset: $DATASET"
echo "Training seed: 1"

PYTHONPATH=. python -u experiments/train.py \
    "$MODEL" shortest_path "$EXPERIMENT_TAG"

echo "Training complete: $MODEL / $EXPERIMENT_TAG"