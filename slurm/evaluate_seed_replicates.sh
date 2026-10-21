#!/bin/bash
#SBATCH --job-name=eval_seeds
#SBATCH --partition=beards
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-17%3
#SBATCH --output=slurm/eval_seeds_%A_%a.out
#SBATCH --error=slurm/eval_seeds_%A_%a.err

set -e

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

# Select training replication
VERSION=${1:?Please provide version 1, 2, 3, 4, or 5}

if [[ ! "$VERSION" =~ ^[1-5]$ ]]; then
    echo "ERROR: Version must be 1, 2, 3, 4, or 5."
    exit 1
fi

# Six training distributions
DATASETS=(
    "onein_reps100"
    "onein_filtered"
    "mixed_55_15_reps100"
    "mixed_60_10_reps100"
    "mixed_55_15_filtered_reps100"
    "mixed_filtered"
)

# Three architectures
MODELS=(
    "tropical_v2"
    "edge_transformer"
    "gnn"
)

# Assign one model/dataset combination to each task
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

# Seed 1 uses the original experiment tag without a version suffix
if [[ "$VERSION" == "1" ]]; then
    EXPERIMENT_TAG="${DATASET}"
else
    EXPERIMENT_TAG="${DATASET}_v${VERSION}"
fi


echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Training Dataset: $DATASET"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "========================================"

# UNFILTERED SYNTHETIC EVALUATIONS

for EVAL_MODE in id_new value_ood ood_size; do

    echo "Evaluating $MODEL on UNFILTERED $EVAL_MODE"

    PYTHONPATH=. python -u evaluation/evaluate.py \
        "$MODEL" shortest_path "$EVAL_MODE" \
        "$EXPERIMENT_TAG" eval_unfiltered

done

# FILTERED SYNTHETIC EVALUATIONS

for EVAL_MODE in id_new value_ood ood_size; do

    echo "Evaluating $MODEL on FILTERED $EVAL_MODE"

    PYTHONPATH=. python -u evaluation/evaluate.py \
        "$MODEL" shortest_path "$EVAL_MODE" \
        "$EXPERIMENT_TAG" eval_filtered

done

# WOOD BENCHMARK

echo "Evaluating $MODEL on WOOD"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood "$EXPERIMENT_TAG"

# EXTERNAL NETWORKS

echo "Evaluating $MODEL on EXTERNAL"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path external "$EXPERIMENT_TAG"

echo "========================================"
echo "Completed all evaluations for:"
echo "$MODEL / $EXPERIMENT_TAG"
echo "========================================"
