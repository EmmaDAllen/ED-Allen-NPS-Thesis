#!/bin/bash
#SBATCH --job-name=baseline_mincost
#SBATCH --partition=beards
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/baseline_mincost_%j.out
#SBATCH --error=logs/baseline_mincost_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

echo "========================================"
echo "Evaluating Transformer - Min Cost Flow"
echo "========================================"
PYTHONPATH=. python evaluation/evaluate.py transformer min_cost_flow ood_size

echo "========================================"
echo "Evaluating Edge Transformer - Min Cost Flow"
echo "========================================"
PYTHONPATH=. python evaluation/evaluate.py edge_transformer min_cost_flow ood_size

echo "========================================"
echo "Evaluating GNN - Min Cost Flow"
echo "========================================"
PYTHONPATH=. python evaluation/evaluate.py gnn min_cost_flow ood_size

echo "========================================"
echo "All Min Cost Flow baseline evaluations complete"
echo "========================================"
