#!/bin/bash
#SBATCH --job-name=tropical_maxflow
#SBATCH --partition=beards
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/tropical_maxflow_%j.out
#SBATCH --error=logs/tropical_maxflow_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python -u evaluation/evaluate.py tropical max_flow ood_size
