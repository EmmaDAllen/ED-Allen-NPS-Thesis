#!/bin/bash
#SBATCH --job-name=eval_sensitivity
#SBATCH --partition=beards
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-29%6
#SBATCH --output=logs/eval_sensitivity_%A_%a.out
#SBATCH --error=logs/eval_sensitivity_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

# Same experiments and ordering as training.
DATASETS=(
    "mixed_filtered_graphloss"
    "mixed_filtered_normmax"
)

MODELS=("tropical_v2" "edge_transformer" "gnn")

TASK_ID=${SLURM_ARRAY_TASK_ID}

VERSION=$((TASK_ID / 6 + 1))
COMBO_INDEX=$((TASK_ID % 6))

DATASET_INDEX=$((COMBO_INDEX / 3))
MODEL_INDEX=$((COMBO_INDEX % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

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

# Verify that training produced the correct checkpoint.
CHECKPOINT="saved_models/${MODEL}_shortest_path_${EXPERIMENT_TAG}_best_model.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Missing checkpoint: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint found: $CHECKPOINT"

# Three unfiltered synthetic evaluation sets.
for EVAL_MODE in id_new value_ood ood_size; do
    echo "UNFILTERED: ${EVAL_MODE}"

    PYTHONPATH=. python -u evaluation/evaluate.py \
        "$MODEL" shortest_path "$EVAL_MODE" \
        "$EXPERIMENT_TAG" eval_unfiltered
done

# Three filtered synthetic evaluation sets.
for EVAL_MODE in id_new value_ood ood_size; do
    echo "FILTERED: ${EVAL_MODE}"

    PYTHONPATH=. python -u evaluation/evaluate.py \
        "$MODEL" shortest_path "$EVAL_MODE" \
        "$EXPERIMENT_TAG" eval_filtered
done

# Original Wood benchmark.
echo "WOOD"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood "$EXPERIMENT_TAG"

# Modified Wood penalty distribution.
echo "WOOD EXP"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood_exp "$EXPERIMENT_TAG"

# External real-world networks.
echo "EXTERNAL"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path external "$EXPERIMENT_TAG"

echo "========================================"
echo "All evaluations completed successfully."
echo "Model: ${MODEL}"
echo "Experiment: ${EXPERIMENT_TAG}"
echo "========================================"
