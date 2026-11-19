#!/bin/bash
#SBATCH --job-name=train_sensitivity
#SBATCH --partition=beards
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-29%6
#SBATCH --output=logs/train_sensitivity_%A_%a.out
#SBATCH --error=logs/train_sensitivity_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

# Two sensitivity experiments.
DATASETS=(
    "mixed_filtered_graphloss"
    "mixed_filtered_normmax"
)

# Three final model architectures.
MODELS=("tropical_v2" "edge_transformer" "gnn")

# Each seed contains 2 experiments x 3 models = 6 tasks.
TASK_ID=${SLURM_ARRAY_TASK_ID}

VERSION=$((TASK_ID / 6 + 1))
COMBO_INDEX=$((TASK_ID % 6))

DATASET_INDEX=$((COMBO_INDEX / 3))
MODEL_INDEX=$((COMBO_INDEX % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

# Seed 1 uses the base experiment tag.
# Seeds 2-5 append _v2 through _v5.
if [[ "$VERSION" -eq 1 ]]; then
    EXPERIMENT_TAG="$DATASET"
else
    EXPERIMENT_TAG="${DATASET}_v${VERSION}"
fi

echo "========================================"
echo "Job: ${SLURM_JOB_ID}"
echo "Task: ${TASK_ID}"
echo "Training seed: ${VERSION}"
echo "Experiment: ${DATASET}"
echo "Model: ${MODEL}"
echo "Experiment tag: ${EXPERIMENT_TAG}"
echo "========================================"

# Train on the corrected mixed-filtered dataset.
# train.py selects the appropriate loss and normalization.
PYTHONPATH=. python -u experiments/train.py \
    "$MODEL" shortest_path "$EXPERIMENT_TAG"

echo "Training complete: ${MODEL} / ${EXPERIMENT_TAG}"
