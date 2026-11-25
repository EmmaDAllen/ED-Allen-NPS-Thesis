#!/bin/bash
#SBATCH --job-name=train_wood_6000
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --array=0-14%6
#SBATCH --output=logs/train_wood_6000_%A_%a.out
#SBATCH --error=logs/train_wood_6000_%A_%a.err
set -euo pipefail
cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate
DATASET=data/training_data_shortest_path_wood_only_6000.json
[[ -f "$DATASET" ]] || { echo "Missing $DATASET" >&2; exit 1; }
MODELS=(tropical_v2 edge_transformer gnn)
TASK_ID=${SLURM_ARRAY_TASK_ID}
VERSION=$(( TASK_ID / 3 + 1 ))
MODEL=${MODELS[$(( TASK_ID % 3 ))]}
if [[ "$VERSION" -eq 1 ]]; then TAG=wood_only_6000; else TAG="wood_only_6000_v${VERSION}"; fi
echo "Model=$MODEL tag=$TAG training_seed=$VERSION"
PYTHONPATH=. python -u experiments/train.py "$MODEL" shortest_path "$TAG"