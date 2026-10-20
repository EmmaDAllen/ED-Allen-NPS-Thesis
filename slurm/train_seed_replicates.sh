#!/bin/bash
#SBATCH --job-name=train_seeds
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-17%3
#SBATCH --output=slurm/train_seeds_%A_%a.out
#SBATCH --error=slurm/train_seeds_%A_%a.err

set -e

# Navigate to thesis repository
cd ~/ED-Allen-NPS-Thesis

# Activate thesis environment
source ~/thesis/bin/activate

# Select training seed from command-line argument
VERSION=${1:?Please provide a version number (2, 3, 4, or 5)}

if [[ ! "$VERSION" =~ ^[2-5]$ ]]; then
    echo "ERROR: Version must be 2, 3, 4, or 5."
    exit 1
fi

# Six existing training distributions
DATASETS=(
    "onein_reps100"
    "onein_filtered"
    "mixed_55_15_reps100"
    "mixed_60_10_reps100"
    "mixed_55_15_filtered_reps100"
    "mixed_filtered"
)

# Three selected architectures
MODELS=(
    "tropical_v2"
    "edge_transformer"
    "gnn"
)

# Assign one dataset/model combination to each array task
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

# Append version suffix to the experiment tag
EXPERIMENT_TAG="${DATASET}_v${VERSION}"

echo "========================================"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Experiment Tag: ${EXPERIMENT_TAG}"
echo "Training Seed: ${VERSION}"
echo "========================================"

# Run training
PYTHONPATH=. python -u experiments/train.py \
    "$MODEL" shortest_path "$EXPERIMENT_TAG"

echo "Completed training: ${MODEL} / ${EXPERIMENT_TAG}"
