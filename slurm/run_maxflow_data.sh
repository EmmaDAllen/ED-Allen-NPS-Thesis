#!/bin/bash
#SBATCH --job-name=maxflow_data
#SBATCH --partition=beards
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/maxflow_data_%j.out
#SBATCH --error=logs/maxflow_data_%j.err

source ~/.bashrc
conda activate thesis

cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python data/generate_data.py max_flow
