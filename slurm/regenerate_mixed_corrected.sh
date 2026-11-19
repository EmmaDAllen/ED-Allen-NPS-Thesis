#!/bin/bash
#SBATCH --job-name=regen_mixed
#SBATCH --partition=beards
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=regen_mixed_%A_%a.out
#SBATCH --error=regen_mixed_%A_%a.err

set -euo pipefail

# Move to the repository directory.
cd "$SLURM_SUBMIT_DIR"

# Four mixed-topology training distributions.
DATASETS=(
    "mixed_55_15_reps100"
    "mixed_60_10_reps100"
    "mixed_55_15_filtered_reps100"
    "mixed_filtered"
)

# Select the dataset corresponding to this array task.
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Save corrected training data in the repository root.
OUTPUT="training_data_shortest_path_${DATASET}_corrected.json"

echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "=========================================="

# Regenerate graphs and solve MIP labels using the corrected generator.
python -u data/regenerate_corrected.py "$DATASET" "$OUTPUT"

echo "=========================================="
echo "Completed dataset: ${DATASET}"
echo "Finish time: $(date)"
echo "=========================================="
