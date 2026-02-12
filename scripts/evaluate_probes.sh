#!/bin/bash
set -euo pipefail

# Submit probe evaluation jobs for all model/dataset/bias combinations

models=(qwen-3-8b llama-3.1-8b gemma-3-4b) #qwen-3-8b llama-3.1-8b gemma-3-4b)
datasets=(arc-challenge mmlu aqua commonsense_qa)
biases=(expert self metadata) #expert self metadata)

# Probe type: bias, has-switched, or will-switch
PROBE=${PROBE:-has-switched}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Submitting probe evaluation jobs for all combinations..."
echo "Probe: ${PROBE}"
echo "Models: ${models[*]}"
echo "Datasets: ${datasets[*]}"
echo "Biases: ${biases[*]}"
echo "Total jobs: $((${#models[@]} * ${#datasets[@]} * ${#biases[@]}))"
echo ""

job_count=0
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for bias in "${biases[@]}"; do
      echo "[$((++job_count))] Submitting: probe=$PROBE model=$model dataset=$dataset bias=$bias"
      TIME=04:00:00 PROBE=$PROBE MODEL=$model DATASET=$dataset BIAS=$bias ./scripts/submit_eval_probes.sh
      sleep 1  # Small delay between submissions
    done
  done
done

echo ""
echo "Submitted $job_count jobs total."
echo "Monitor with: squeue -u \$USER"
