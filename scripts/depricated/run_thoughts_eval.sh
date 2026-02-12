#!/bin/bash
# filepath: run_thoughts_eval.sh

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=thoughts_eval
#SBATCH --output=thoughts_eval_%j.out
#SBATCH --error=thoughts_eval_%j.err

source ~/.bashrc
conda activate "${CONDA_ENV}"

python thoughts_eval.py --model qwen-2.5-7b --data mmlu  --use_cot --give_hint --n_ckpts 5 --predict hint --ckpt suffix
