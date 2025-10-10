#!/bin/bash
set -euo pipefail

source "$HOME/miniforge/etc/profile.d/conda.sh"
conda activate skyline

models=(gemma-3-4b llama-3.1-8b qwen-3-8b)
# datasets=(aqua mmlu arc-challenge commonsense_qa)
datasets=(aqua mmlu arc-challenge commonsense_qa)

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running ${model} on ${dataset}"
    python thoughts_eval.py --model "${model}" --dataset "${dataset}" --evaluate
  done
done
