#!/bin/bash
#SBATCH --job-name=eval_mix_id
#SBATCH --partition=beards
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/eval_mixed_topology_id_%j.out
#SBATCH --error=logs/eval_mixed_topology_id_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

nvidia-smi

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new mixed_topology
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new mixed_topology
