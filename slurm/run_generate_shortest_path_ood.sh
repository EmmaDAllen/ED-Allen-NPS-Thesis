#!/bin/bash

#SBATCH --job-name=gen_sp_ood
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/gen_sp_ood_%j.out
#SBATCH --error=logs/gen_sp_ood_%j.err

source ~/thesis/bin/activate

cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python -u evaluation/generate_evaluation_graphs.py shortest_path ood_size
