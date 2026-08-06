#!/bin/bash
#SBATCH --job-name=tropical_ood
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/tropical_ood_%j.out
#SBATCH --error=logs/tropical_ood_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python evaluation/evaluate.py tropical shortest_path ood_size
