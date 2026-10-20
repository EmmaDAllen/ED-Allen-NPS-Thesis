#!/bin/bash
#SBATCH --job-name=eval_55_15_filt_u
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/eval_55_15_filt_unfiltered_%j.out
#SBATCH --error=logs/eval_55_15_filt_unfiltered_%j.err

set -e
source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis
nvidia-smi

echo "mixed_55_15_filtered_reps100 -> unfiltered id_new"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new mixed_55_15_filtered_reps100 eval_unfiltered

echo "mixed_55_15_filtered_reps100 -> unfiltered value_ood"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path value_ood mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path value_ood mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path value_ood mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path value_ood mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path value_ood mixed_55_15_filtered_reps100 eval_unfiltered

echo "mixed_55_15_filtered_reps100 -> unfiltered ood_size"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size mixed_55_15_filtered_reps100 eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size mixed_55_15_filtered_reps100 eval_unfiltered

echo "mixed_55_15_filtered_reps100 -> unfiltered wood"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path wood mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path wood mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path wood mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path wood mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path wood mixed_55_15_filtered_reps100 candidate4

echo "mixed_55_15_filtered_reps100 -> unfiltered external"
PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path external mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path external mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path external mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path external mixed_55_15_filtered_reps100 candidate4
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path external mixed_55_15_filtered_reps100 candidate4


