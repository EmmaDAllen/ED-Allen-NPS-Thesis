#!/bin/bash
#SBATCH --job-name=all_ood_eval
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/all_ood_eval_%j.out
#SBATCH --error=logs/all_ood_eval_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python evaluation/evaluate.py tropical shortest_path ood_size
PYTHONPATH=. python evaluation/evaluate.py tropical_v2 shortest_path ood_size
PYTHONPATH=. python evaluation/evaluate.py transformer shortest_path ood_size
PYTHONPATH=. python evaluation/evaluate.py edge_transformer shortest_path ood_size
PYTHONPATH=. python evaluation/evaluate.py gnn shortest_path ood_size
