#!/bin/bash
set -euo pipefail

# Submit LLM baseline evaluation jobs for all model/dataset/bias combinations

models=(qwen-3-8b) # llama-3.1-8b gemma-3-4b)
datasets=(mmlu) # mmlu aqua) # arc-challenge commonsense_qa mmlu aqua)
biases=(self)

# LLM model for baseline evaluation
LLM=${LLM:-gpt-5-mini}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Submitting LLM baseline evaluation jobs for all combinations..."
echo "LLM: ${LLM}"
echo "Models: ${models[*]}"
echo "Datasets: ${datasets[*]}"
echo "Biases: ${biases[*]}"
echo "Total jobs: $((${#models[@]} * ${#datasets[@]} * ${#biases[@]}))"
echo ""

job_count=0
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for bias in "${biases[@]}"; do
      echo "[$((++job_count))] Submitting: llm=$LLM model=$model dataset=$dataset bias=$bias"
      TIME=24:00:00 LLM=$LLM MODEL=$model DATASET=$dataset BIAS=$bias ./scripts/submit_eval_llm.sh
      sleep 1  # Small delay between submissions
    done
  done
done

echo ""
echo "Submitted $job_count jobs total."
echo "Monitor with: squeue -u \$USER"
