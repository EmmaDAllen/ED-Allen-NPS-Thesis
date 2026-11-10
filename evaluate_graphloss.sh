#!/bin/bash
#SBATCH --job-name=eval_graphloss
#SBATCH --partition=beards
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-2%3
#SBATCH --output=slurm/eval_graphloss_%A_%a.out
#SBATCH --error=slurm/eval_graphloss_%A_%a.err

set -e

# Activate thesis environment
source ~/thesis/bin/activate

# Navigate to thesis repository
cd ~/ED-Allen-NPS-Thesis

# Three architectures for graph-balanced loss experiment
MODELS=(
    "tropical_v2"
    "edge_transformer"
    "gnn"
)

# Each array task evaluates one model
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Fixed experiment tag (seed 1)
EXPERIMENT_TAG="mixed_filtered_graphloss"

echo "========================================"
echo "SLURM Job: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "========================================"

# Verify that training checkpoint exists
CHECKPOINT="saved_models/${MODEL}_shortest_path_${EXPERIMENT_TAG}_best_model.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

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

# ORIGINAL WOOD BENCHMARK

echo "Evaluating $MODEL on WOOD"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood "$EXPERIMENT_TAG"

# WOOD PENALTY SENSITIVITY EXPERIMENT

echo "Evaluating $MODEL on WOOD EXP"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood_exp "$EXPERIMENT_TAG"

# EXTERNAL NETWORKS

echo "Evaluating $MODEL on EXTERNAL"

PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path external "$EXPERIMENT_TAG"

echo "========================================"
echo "Completed all evaluations for:"
echo "$MODEL / $EXPERIMENT_TAG"
echo "========================================"

