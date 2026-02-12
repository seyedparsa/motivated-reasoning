#!/bin/bash

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=train_probes
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${SCRIPT_DIR}/.."

python main.py --train_probes --evaluate_probes --model gemma-3-4b --dataset commonsense_qa --bias expert --probe will-switch --n_ckpts 3
