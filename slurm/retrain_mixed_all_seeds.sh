#!/bin/bash
#SBATCH --job-name=retrain_mixed
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-59%6
#SBATCH --output=logs/retrain_mixed_%A_%a.out
#SBATCH --error=logs/retrain_mixed_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

# Four regenerated mixed-topology training distributions.
DATASETS=(
    "mixed_55_15_reps100_corrected"
    "mixed_60_10_reps100_corrected"
    "mixed_55_15_filtered_reps100_corrected"
    "mixed_filtered_corrected"
)

MODELS=("tropical_v2" "edge_transformer" "gnn")

# 12 model/dataset combinations per seed; 5 seeds = 60 array tasks.
TASK_ID=${SLURM_ARRAY_TASK_ID}
VERSION=$((TASK_ID / 12 + 1))
COMBO_INDEX=$((TASK_ID % 12))
DATASET_INDEX=$((COMBO_INDEX / 3))
MODEL_INDEX=$((COMBO_INDEX % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

# Retain the exact existing checkpoint tag naming convention.
if [[ "$VERSION" -eq 1 ]]; then
    EXPERIMENT_TAG="$DATASET"
else
    EXPERIMENT_TAG="${DATASET}_v${VERSION}"
fi

echo "Job: ${SLURM_JOB_ID} | Task: ${TASK_ID}"
echo "Training seed/version: ${VERSION}"
echo "Dataset: ${DATASET} | Model: ${MODEL}"
echo "Experiment tag: ${EXPERIMENT_TAG}"

PYTHONPATH=. python -u experiments/train.py \
    "$MODEL" shortest_path "$EXPERIMENT_TAG"

echo "Training complete: ${MODEL} / ${EXPERIMENT_TAG}"
