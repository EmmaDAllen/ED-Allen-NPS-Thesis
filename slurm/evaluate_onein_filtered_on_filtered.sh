#!/bin/bash
#SBATCH --job-name=eval_1in_filt
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=eval_1in_filt_%j.out
#SBATCH --error=eval_1in_filt_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis


echo "FILTERED ONE-IN -> FILTERED ID"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new onein_filtered eval_filtered


echo "FILTERED ONE-IN -> FILTERED VALUE OOD"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path value_ood onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path value_ood onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path value_ood onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path value_ood onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path value_ood onein_filtered eval_filtered


echo "FILTERED ONE-IN -> FILTERED SIZE OOD"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size onein_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size onein_filtered eval_filtered
