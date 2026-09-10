#!/bin/bash

#SBATCH --job-name=wood_all
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/wood_all_%j.out
#SBATCH --error=logs/wood_all_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

nvidia-smi
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path wood filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path wood filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path wood filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path wood filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path wood filtered
