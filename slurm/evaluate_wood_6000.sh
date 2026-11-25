#!/bin/bash
#SBATCH --job-name=eval_wood_6000
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-29%6
#SBATCH --output=logs/eval_wood_6000_%A_%a.out
#SBATCH --error=logs/eval_wood_6000_%A_%a.err
set -euo pipefail
cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate
MODELS=(tropical_v2 edge_transformer gnn)
EVAL_MODES=(wood wood_exp)
TASK_ID=${SLURM_ARRAY_TASK_ID}
VERSION=$(( TASK_ID / 6 + 1 ))
COMBO=$(( TASK_ID % 6 ))
MODEL=${MODELS[$(( COMBO / 2 ))]}
EVAL_MODE=${EVAL_MODES[$(( COMBO % 2 ))]}
if [[ "$VERSION" -eq 1 ]]; then TAG=wood_only_6000; else TAG="wood_only_6000_v${VERSION}"; fi
CHECKPOINT="saved_models/${MODEL}_shortest_path_${TAG}_best_model.pt"
GRAPH_FILE="evaluation_graphs/shortest_path_${EVAL_MODE}_graphs.pkl"
[[ -f "$CHECKPOINT" ]] || { echo "Missing $CHECKPOINT" >&2; exit 1; }
[[ -f "$GRAPH_FILE" ]] || { echo "Missing $GRAPH_FILE" >&2; exit 1; }
echo "Model=$MODEL tag=$TAG eval_mode=$EVAL_MODE"
PYTHONPATH=. python -u evaluation/evaluate.py "$MODEL" shortest_path "$EVAL_MODE" "$TAG"