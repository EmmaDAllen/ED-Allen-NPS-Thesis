#!/bin/bash
#SBATCH --job-name=eval_mix_v2_ood_c4
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/eval_mix_v2_ood_c4_%j.out
#SBATCH --error=logs/eval_mix_v2_ood_c4_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

nvidia-smi

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size mixed_topology_v2 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size mixed_topology_v2 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size mixed_topology_v2 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size mixed_topology_v2 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size mixed_topology_v2 candidate4
