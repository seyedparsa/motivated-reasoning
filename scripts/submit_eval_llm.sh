#!/bin/bash
set -euo pipefail

# Submit LLM baseline evaluation job

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-24:00:00}
MEM=${MEM:-32g}
GPUS=${GPUS:-0}
CPUS=${CPUS:-1}
NODES=${NODES:-1}

# Default test configuration
MODEL=${MODEL:-qwen-3-8b}
DATASET=${DATASET:-mmlu}
BIAS=${BIAS:-expert}
LLM=${LLM:-gpt-5-nano}

job_name="eval_llm_${MODEL}_${DATASET}_${BIAS}"

echo "Submitting eval LLM job: ${job_name}"
echo "Model: ${MODEL}, Dataset: ${DATASET}, Bias: ${BIAS}, LLM: ${LLM}"

sbatch --export=ALL <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err

set -euo pipefail

# Load environment variables from .env
source ${SCRIPT_DIR}/../.env

# Activate conda environment
source ~/.bashrc
conda activate \${CONDA_ENV}

# Set environment variables
export HF_HOME
export MOTIVATION_HOME
export HF_TOKEN
export HF_USE_SOFTFILELOCK=1
export PYTHONUNBUFFERED=1

echo "=== LLM Baseline Evaluation ==="
echo "Job: ${job_name}"
echo "Model: ${MODEL} Dataset: ${DATASET} Bias: ${BIAS} LLM: ${LLM}"
echo "Git commit: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

echo "Evaluating LLM baseline (${LLM})..."
python main.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --bias "${BIAS}" \
  --evaluate_llm \
  --llm "${LLM}"
echo "Done!"
EOF
