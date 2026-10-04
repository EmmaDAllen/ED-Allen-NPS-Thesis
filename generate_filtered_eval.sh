#!/bin/bash
#SBATCH --job-name=gen_filtered_eval
#SBATCH --partition=beards
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=gen_filtered_eval_%j.out
#SBATCH --error=gen_filtered_eval_%j.err

source ~/thesis/bin/activate

cd ~/ED-Allen-NPS-Thesis

echo "Generating filtered ID evaluation graphs..."
PYTHONPATH=. python -u evaluation/generate_filtered_evaluation_graphs.py shortest_path id_new

echo "Generating filtered value OOD evaluation graphs..."
PYTHONPATH=. python -u evaluation/generate_filtered_evaluation_graphs.py shortest_path value_ood

echo "Generating filtered size OOD evaluation graphs..."
PYTHONPATH=. python -u evaluation/generate_filtered_evaluation_graphs.py shortest_path ood_size

echo "All filtered evaluation graph generation complete."
