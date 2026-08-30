#!/bin/bash

#SBATCH --job-name=evaluate_filtered
#SBATCH --partition=beards
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/evaluate_filtered_%j.out
#SBATCH --error=logs/evaluate_filtered_%j.err

source ~/thesis/bin/activate

cd ~/ED-Allen-NPS-Thesis

nvidia-smi

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size filtered
