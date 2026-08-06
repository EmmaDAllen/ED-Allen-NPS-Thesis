#!/bin/bash
#SBATCH --job-name=transformer_ood
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/transformer_ood_%j.out
#SBATCH --error=logs/transformer_ood_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python evaluation/evaluate.py transformer shortest_path ood_size
