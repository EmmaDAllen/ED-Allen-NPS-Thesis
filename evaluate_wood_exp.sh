#!/bin/bash
#SBATCH --job-name=wood_exp
#SBATCH --partition=beards
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-17%3
#SBATCH --output=slurm/wood_exp_%A_%a.out
#SBATCH --error=slurm/wood_exp_%A_%a.err

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

# Assign one model/dataset combination to each array task
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

DATASET=${DATASETS[$DATASET_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

# Seed 1 uses the original experiment tag
# Seeds 2-5 use version suffixes
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
echo "Training Version: $VERSION"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Evaluation Mode: wood_exp"
echo "========================================"

# Verify the new Wood evaluation graphs exist
GRAPH_FILE="evaluation_graphs/shortest_path_wood_exp_graphs.pkl"

if [[ ! -f "$GRAPH_FILE" ]]; then
    echo "ERROR: Missing evaluation graphs: $GRAPH_FILE"
    exit 1
fi

# Verify the trained checkpoint exists
CHECKPOINT="saved_models/${MODEL}_shortest_path_${EXPERIMENT_TAG}_best_model.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Missing checkpoint: $CHECKPOINT"
    exit 1
fi

# WOOD PENALTY SENSITIVITY EVALUATION

echo "Evaluating $MODEL on WOOD_EXP"
echo "Checkpoint: $CHECKPOINT"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood_exp "$EXPERIMENT_TAG"

echo "========================================"
echo "Completed Wood sensitivity evaluation:"
echo "$MODEL / $EXPERIMENT_TAG"
echo "========================================"
