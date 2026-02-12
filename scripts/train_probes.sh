#!/bin/bash
set -euo pipefail

# Submit probe training jobs for all model/dataset/bias/probe combinations
# Usage:
#   ./scripts/train_probes.sh                    # Run all combinations
#   PROBE=mot_vs_alg ./scripts/train_probes.sh   # Run specific probe type

models=(qwen-3-8b llama-3.1-8b gemma-3-4b)
datasets=(arc-challenge mmlu aqua commonsense_qa)
biases=(expert self metadata)
probes=(hint-recovery mot_vs_alg mot_vs_res mot_vs_oth)

# Allow overriding specific values
MODELS=${MODELS:-${models[@]}}
DATASETS=${DATASETS:-${datasets[@]}}
BIASES=${BIASES:-${biases[@]}}
PROBES=${PROBES:-${probes[@]}}

# Convert to arrays if provided as space-separated strings
read -ra MODELS <<< "${MODELS[*]}"
read -ra DATASETS <<< "${DATASETS[*]}"
read -ra BIASES <<< "${BIASES[*]}"
read -ra PROBES <<< "${PROBES[*]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Submitting probe training jobs for all combinations..."
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Biases: ${BIASES[*]}"
echo "Probes: ${PROBES[*]}"
echo "Total jobs: $((${#MODELS[@]} * ${#DATASETS[@]} * ${#BIASES[@]} * ${#PROBES[@]}))"
echo ""

job_count=0
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for bias in "${BIASES[@]}"; do
      for probe in "${PROBES[@]}"; do
        echo "[$((++job_count))] Submitting: model=$model dataset=$dataset bias=$bias probe=$probe"
        MODEL=$model DATASET=$dataset BIAS=$bias PROBE=$probe ./scripts/submit_probes.sh
        sleep 1  # Small delay between submissions
      done
    done
  done
done

echo ""
echo "Submitted $job_count jobs total."
echo "Monitor with: squeue -u \$USER"
