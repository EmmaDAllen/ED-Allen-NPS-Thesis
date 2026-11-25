#!/bin/bash
#SBATCH --job-name=gen_wood_6000
#SBATCH --partition=beards
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=logs/gen_wood_6000_%j.out
#SBATCH --error=logs/gen_wood_6000_%j.err
set -euo pipefail
cd ~/ED-Allen-NPS-Thesis
source ~/thesis/bin/activate
PYTHONPATH=. python -u data/generate_wood_only_6000.py