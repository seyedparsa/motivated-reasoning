#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=test_cross_dataset
#SBATCH --output=test_cross_dataset_%j.out
#SBATCH --error=test_cross_dataset_%j.err

set -euo pipefail
cd ~/neural_controllers
source .env
source ~/.bashrc
conda activate ${CONDA_ENV}

export HF_HOME
export MOTIVATION_HOME
export HF_TOKEN
export HF_USE_SOFTFILELOCK=1
export PYTHONUNBUFFERED=1

echo "=== Standard eval (train=eval=mmlu/expert) ==="
python main.py --model qwen-3-8b --dataset mmlu --bias expert --probe mot_vs_alg \
    --evaluate_probes --n_ckpts 3 --ckpt rel --scale small

echo ""
echo "=== Cross-dataset eval (train=mmlu/expert, eval=arc-challenge/expert) ==="
python main.py --model qwen-3-8b --dataset mmlu --bias expert --probe mot_vs_alg \
    --evaluate_probes --eval_dataset arc-challenge --n_ckpts 3 --ckpt rel --scale small

echo "Done\!"
