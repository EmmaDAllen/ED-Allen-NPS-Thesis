#!/bin/bash
#SBATCH --job-name=eval_mix_wood
#SBATCH --partition=beards
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/eval_mixed_topology_wood_%j.out
#SBATCH --error=logs/eval_mixed_topology_wood_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

nvidia-smi

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path wood mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path wood mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path wood mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path wood mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path wood mixed_topology
