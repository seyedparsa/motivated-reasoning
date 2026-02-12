#!/bin/bash
set -euo pipefail

# Submit probe jobs for all model/dataset/bias combinations
# This will train linear probes alongside existing RFM probes

models=(qwen-3-8b llama-3.1-8b gemma-3-4b)
datasets=(arc-challenge mmlu aqua commonsense_qa)
biases=(expert self metadata)

# Probe type: bias, has-switched, or will-switch
PROBE=${PROBE:-bias}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Submitting ${PROBE} probe jobs for all combinations..."
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
      TIME=01:00:00 PROBE=$PROBE MODEL=$model DATASET=$dataset BIAS=$bias ./scripts/submit_probe.sh
      sleep 1  # Small delay between submissions
    done
  done
done

echo ""
echo "Submitted $job_count jobs total."
echo "Monitor with: squeue -u \$USER"
