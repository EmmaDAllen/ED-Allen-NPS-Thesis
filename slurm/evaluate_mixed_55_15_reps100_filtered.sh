#!/bin/bash
#SBATCH --job-name=eval_55_15_f
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/eval_55_15_filtered_%j.out
#SBATCH --error=logs/eval_55_15_filtered_%j.err

set -e
source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis
nvidia-smi

echo "mixed_55_15_reps100 -> filtered id_new"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new mixed_55_15_reps100 eval_filtered

echo "mixed_55_15_reps100 -> filtered value_ood"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path value_ood mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path value_ood mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path value_ood mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path value_ood mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path value_ood mixed_55_15_reps100 eval_filtered

echo "mixed_55_15_reps100 -> filtered ood_size"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size mixed_55_15_reps100 eval_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size mixed_55_15_reps100 eval_filtered


