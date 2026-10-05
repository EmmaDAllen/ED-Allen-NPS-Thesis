#!/bin/bash
#SBATCH --job-name=eval_filt_unfilt
#SBATCH --partition=beards
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=eval_filt_unfilt_%j.out
#SBATCH --error=eval_filt_unfilt_%j.err

source ~/thesis/bin/activate
cd ~/ED-Allen-NPS-Thesis

echo "FILTERED ONE-IN -> UNFILTERED ID"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path id_new onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path id_new onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path id_new onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path id_new onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path id_new onein_filtered eval_unfiltered


echo "FILTERED ONE-IN -> UNFILTERED VALUE OOD"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path value_ood onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path value_ood onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path value_ood onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path value_ood onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path value_ood onein_filtered eval_unfiltered


echo "FILTERED ONE-IN -> UNFILTERED SIZE OOD"


PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path ood_size onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path ood_size onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path ood_size onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path ood_size onein_filtered eval_unfiltered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path ood_size onein_filtered eval_unfiltered


echo "FILTERED ONE-IN -> WOOD"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path wood onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path wood onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path wood onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path wood onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path wood onein_filtered


echo "FILTERED ONE-IN -> EXTERNAL"

PYTHONPATH=. python -u evaluation/evaluate.py tropical shortest_path external onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py tropical_v2 shortest_path external onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py transformer shortest_path external onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py edge_transformer shortest_path external onein_filtered
PYTHONPATH=. python -u evaluation/evaluate.py gnn shortest_path external onein_filtered
