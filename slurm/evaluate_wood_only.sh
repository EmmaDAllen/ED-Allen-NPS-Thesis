#!/bin/bash
#SBATCH --job-name=eval_wood_only
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-8%2
#SBATCH --output=logs/eval_wood_only_%A_%a.out
#SBATCH --error=logs/eval_wood_only_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

MODELS=("tropical_v2" "edge_transformer" "gnn")
EVAL_MODES=("wood" "wood_exp" "external")
EXPERIMENT_TAG="wood_only"

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_INDEX=$((TASK_ID / 3))
EVAL_INDEX=$((TASK_ID % 3))
MODEL=${MODELS[$MODEL_INDEX]}
EVAL_MODE=${EVAL_MODES[$EVAL_INDEX]}

CHECKPOINT="saved_models/${MODEL}_shortest_path_${EXPERIMENT_TAG}_best_model.pt"
GRAPH_FILE="evaluation_graphs/shortest_path_${EVAL_MODE}_graphs.pkl"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -f "$GRAPH_FILE" ]] || { echo "Missing evaluation graphs: $GRAPH_FILE" >&2; exit 1; }

# evaluate.py selects 10/10 normalization for wood_only.
echo "Job ${SLURM_JOB_ID}, task ${TASK_ID}: ${MODEL} / ${EXPERIMENT_TAG} / ${EVAL_MODE}"
PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path "$EVAL_MODE" "$EXPERIMENT_TAG"
echo "Evaluation complete: ${MODEL} / ${EXPERIMENT_TAG} / ${EVAL_MODE}"