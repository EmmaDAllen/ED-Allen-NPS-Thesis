#!/bin/bash

#SBATCH --job-name=train_v2_repeats
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/train_v2_repeats_%j.out
#SBATCH --error=logs/train_v2_repeats_%j.err

source ~/thesis/bin/activate

cd ~/ED-Allen-NPS-Thesis

nvidia-smi

PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat1
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat2
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat3
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat4
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat5
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat6
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat7
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat8
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat9
PYTHONPATH=. python -u experiments/train.py tropical_v2 shortest_path v2_repeat10
