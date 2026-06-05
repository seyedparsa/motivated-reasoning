#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=test_permutation
#SBATCH --output=test_permutation_%j.out
#SBATCH --error=test_permutation_%j.err

set -euo pipefail
cd ~/neural_controllers
source .env
source ~/.bashrc
conda activate ${CONDA_ENV}

export HF_HOME MOTIVATION_HOME HF_TOKEN HF_USE_SOFTFILELOCK=1 PYTHONUNBUFFERED=1

echo "=== Cross-dataset with PERMUTED eval labels ==="
python main.py --model qwen-3-8b --dataset mmlu --bias expert --probe mot_vs_alg \
    --evaluate_probes --eval_dataset arc-challenge --n_ckpts 3 --ckpt rel --scale small \
    --permute_eval_labels

echo "Done\!"
