#!/bin/bash
set -euo pipefail

# Submit evaluation jobs for generated responses across models/datasets.

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-04:00:00}
MEM=${MEM:-48g}
GPUS=${GPUS:-0}
CPUS=${CPUS:-4}
NODES=${NODES:-1}
JOB_PREFIX=${JOB_PREFIX:-evalresp}
ACTIVATE=${ACTIVATE:-}
DEBUG=${DEBUG:-0}

models=(qwen-3-8b llama-3.1-8b gemma-3-4b)
datasets=(mmlu aqua commonsense_qa arc-challenge)

submit_job() {
  local model="$1"
  local dataset="$2"
  local safe_model="${model//-/_}"
  safe_model="${safe_model//./_}"
  local safe_dataset="${dataset//-/_}"
  local job_base="${JOB_PREFIX}_${safe_model}_${safe_dataset}"

  echo "Submitting ${job_base}"

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
#SBATCH --job-name=${job_base}
#SBATCH --output=${job_base}_%j.out
#SBATCH --error=${job_base}_%j.err

set -euo pipefail
[[ "${DEBUG}" == "1" ]] && set -x

# Load environment variables from .env
source ${SCRIPT_DIR}/../.env

export HF_HOME
export MOTIVATION_HOME
export HF_TOKEN
export OPENAI_API_KEY
export HF_USE_SOFTFILELOCK=1
export PYTHONUNBUFFERED=1

[[ -n "${ACTIVATE}" ]] && eval "${ACTIVATE}"

echo "Job: ${job_base}"
echo "Model: ${model} Dataset: ${dataset}"
echo "Git commit: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python main.py \
  --evaluate \
  --model "${model}" \
  --dataset "${dataset}"
EOF
}

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    submit_job "${model}" "${dataset}"
  done
done

