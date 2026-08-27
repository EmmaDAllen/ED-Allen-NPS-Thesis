#!/bin/bash
#SBATCH --job-name=train_all
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

nvidia-smi
PYTHONPATH=. python -u experiments/train.py tropical shortest_path retrain_v2
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path retrain_v2
PYTHONPATH=. python -u experiments/train.py transformer shortest_path retrain_v2
PYTHONPATH=. python -u experiments/train.py edge_transformer shortest_path retrain_v2
PYTHONPATH=. python -u experiments/train.py gnn shortest_path retrain_v2
