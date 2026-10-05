#!/bin/bash
#SBATCH --job-name=eval_mix_filt
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=eval_mix_filt_%j.out
#SBATCH --error=eval_mix_filt_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis


echo "FILTERED MIXED -> FILTERED ID"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new mixed_filtered eval_filtered


echo "FILTERED MIXED -> FILTERED VALUE OOD"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path value_ood mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path value_ood mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path value_ood mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path value_ood mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path value_ood mixed_filtered eval_filtered


echo "FILTERED MIXED -> FILTERED SIZE OOD"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size mixed_filtered eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size mixed_filtered eval_filtered
