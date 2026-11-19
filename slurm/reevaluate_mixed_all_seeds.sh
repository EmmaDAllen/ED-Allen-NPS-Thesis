#!/bin/bash
#SBATCH --job-name=reeval_mixed
#SBATCH --partition=beards
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-59%6
#SBATCH --output=logs/reeval_mixed_%A_%a.out
#SBATCH --error=logs/reeval_mixed_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

DATASETS=(
    "mixed_55_15_reps100_corrected"
    "mixed_60_10_reps100_corrected"
    "mixed_55_15_filtered_reps100_corrected"
    "mixed_filtered_corrected"
)
MODELS=("tropical_v2" "edge_transformer" "gnn")

# Same array mapping as training, so each task evaluates its own retrained model.
TASK_ID=${SLURM_ARRAY_TASK_ID}
VERSION=$((TASK_ID / 12 + 1))
COMBO_INDEX=$((TASK_ID % 12))
DATASET_INDEX=$((COMBO_INDEX / 3))
MODEL_INDEX=$((COMBO_INDEX % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

if [[ "$VERSION" -eq 1 ]]; then
    EXPERIMENT_TAG="$DATASET"
else
    EXPERIMENT_TAG="${DATASET}_v${VERSION}"
fi

echo "Job: ${SLURM_JOB_ID} | Task: ${TASK_ID}"
echo "Training seed/version: ${VERSION}"
echo "Dataset: ${DATASET} | Model: ${MODEL}"
echo "Experiment tag: ${EXPERIMENT_TAG}"

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

# Wood and external network benchmarks.
echo "WOOD"
PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood "$EXPERIMENT_TAG"

# WOOD EXPERIMENTAL PENALTY DISTRIBUTION
echo "WOOD EXP"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood_exp "$EXPERIMENT_TAG"

echo "EXTERNAL"
PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path external "$EXPERIMENT_TAG"

echo "Evaluations complete: ${MODEL} / ${EXPERIMENT_TAG}"
