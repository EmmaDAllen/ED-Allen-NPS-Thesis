#!/bin/bash
#SBATCH --job-name=baseline_maxflow
#SBATCH --partition=beards
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/baseline_maxflow_%j.out
#SBATCH --error=logs/baseline_maxflow_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

echo "========================================"
echo "Evaluating Transformer - Max Flow"
echo "========================================"
PYTHONPATH=. python -u  evaluation/evaluate.py transformer max_flow ood_size

echo "========================================"
echo "Evaluating Edge Transformer - Max Flow"
echo "========================================"
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer max_flow ood_size

echo "========================================"
echo "Evaluating GNN - Max Flow"
echo "========================================"
PYTHONPATH=. python -u evaluation/evaluate.py gnn max_flow ood_size

echo "========================================"
echo "All Max Flow baseline evaluations complete"
echo "========================================"
