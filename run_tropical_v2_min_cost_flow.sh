#!/bin/bash
#SBATCH --job-name=tropv2_mincost
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/tropical_v2_mincost_%j.out
#SBATCH --error=logs/tropical_v2_mincost_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

PYTHONPATH=. python evaluation/evaluate.py tropical_v2 min_cost_flow ood_size
