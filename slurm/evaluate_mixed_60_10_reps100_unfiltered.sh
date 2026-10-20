#!/bin/bash
#SBATCH --job-name=eval_60_10_u
#SBATCH --partition=beards
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/eval_60_10_unfiltered_%j.out
#SBATCH --error=logs/eval_60_10_unfiltered_%j.err

set -e
source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis
nvidia-smi

echo "mixed_60_10_reps100 -> unfiltered id_new"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new mixed_60_10_reps100 eval_unfiltered

echo "mixed_60_10_reps100 -> unfiltered value_ood"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path value_ood mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path value_ood mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path value_ood mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path value_ood mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path value_ood mixed_60_10_reps100 eval_unfiltered

echo "mixed_60_10_reps100 -> unfiltered ood_size"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size mixed_60_10_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size mixed_60_10_reps100 eval_unfiltered

echo "mixed_60_10_reps100 -> unfiltered wood"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path wood mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path wood mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path wood mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path wood mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path wood mixed_60_10_reps100 candidate4

echo "mixed_60_10_reps100 -> unfiltered external"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path external mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path external mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path external mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path external mixed_60_10_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path external mixed_60_10_reps100 candidate4


