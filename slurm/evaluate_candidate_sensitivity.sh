#!/bin/bash
#SBATCH --job-name=eval_candidate_sens
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-29%6
#SBATCH --output=slurm/eval_candidate_sens_%A_%a.out
#SBATCH --error=slurm/eval_candidate_sens_%A_%a.err

set -euo pipefail

cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate

# Three model architectures
MODELS=("tropical_v2" "edge_transformer" "gnn")

# Only evaluate K+3 and K+5.
# Existing K+4 results are preserved.
CANDIDATE_EXTRAS=(3 5)

# 3 models x 5 training seeds x 2 candidate sizes = 30 tasks
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CANDIDATE_INDEX=$(((SLURM_ARRAY_TASK_ID / 3) % 2))
SEED=$((SLURM_ARRAY_TASK_ID / 6 + 1))

MODEL=${MODELS[$MODEL_INDEX]}
CANDIDATE_EXTRA=${CANDIDATE_EXTRAS[$CANDIDATE_INDEX]}

# Preserve original checkpoint naming convention
EXPERIMENT_TAG="mixed_filtered_corrected"

if [[ "$SEED" -gt 1 ]]; then
    EXPERIMENT_TAG="${EXPERIMENT_TAG}_v${SEED}"
fi

echo "========================================"
echo "Model: $MODEL"
echo "Training Seed: $SEED"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Candidate Pool: K+$CANDIDATE_EXTRA"
echo "========================================"

# Unfiltered synthetic evaluations
for EVAL_MODE in id_new value_ood ood_size; do
    PYTHONPATH=. python -u evaluation/evaluate.py \
        "$MODEL" shortest_path "$EVAL_MODE" \
        "$EXPERIMENT_TAG" eval_unfiltered "$CANDIDATE_EXTRA"
done

# Filtered synthetic evaluations
for EVAL_MODE in id_new value_ood ood_size; do
    PYTHONPATH=. python -u evaluation/evaluate.py \
        "$MODEL" shortest_path "$EVAL_MODE" \
        "$EXPERIMENT_TAG" eval_filtered "$CANDIDATE_EXTRA"
done

# Wood benchmark
PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood \
    "$EXPERIMENT_TAG" "" "$CANDIDATE_EXTRA"

# Wood experimental benchmark
PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path wood_exp \
    "$EXPERIMENT_TAG" "" "$CANDIDATE_EXTRA"

# External networks
PYTHONPATH=. python -u evaluation/evaluate.py \
    "$MODEL" shortest_path external \
    "$EXPERIMENT_TAG" "" "$CANDIDATE_EXTRA"

echo "========================================"
echo "Completed: $MODEL"
echo "Seed: $SEED"
echo "Candidate: K+$CANDIDATE_EXTRA"
echo "========================================"