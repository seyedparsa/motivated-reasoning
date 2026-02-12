#!/bin/bash
set -euo pipefail

# Submit specific probe evaluation jobs for gemma-3-4b model
# Based on currently running jobs

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-04:00:00}
MEM=${MEM:-100g}
GPUS=${GPUS:-1}
CPUS=${CPUS:-1}
NODES=${NODES:-1}
JOB_PREFIX=${JOB_PREFIX:-probe}
BS_PROBE=${BS_PROBE:-32}
N_CKPTS=${N_CKPTS:-3}
ACTIVATE=${ACTIVATE:-}
DEBUG=${DEBUG:-0}

# Specific job combinations to submit
# Format: model dataset bias probe
jobs=(
  "gemma-3-4b aqua expert bias"
)
#   "gemma-3-4b aqua expert will-switch"
#   "gemma-3-4b aqua metadata bias"
#   "gemma-3-4b aqua metadata has-switched"
#   "gemma-3-4b aqua metadata will-switch"
#   "gemma-3-4b commonsense_qa expert will-switch"
#   "gemma-3-4b commonsense_qa metadata bias"
#   "gemma-3-4b commonsense_qa metadata has-switched"
#   "gemma-3-4b commonsense_qa metadata will-switch"
#   "gemma-3-4b mmlu expert bias"
#   "gemma-3-4b mmlu expert will-switch"
#   "gemma-3-4b mmlu metadata bias"
#   "gemma-3-4b mmlu metadata has-switched"
#   "gemma-3-4b mmlu metadata will-switch"
# )

universal=False

submit_job() {
  local model="$1"
  local dataset="$2"
  local bias="$3"
  local probe="$4"
  local universal="$5"
  local safe_model="${model//-/_}"
  safe_model="${safe_model//./_}"
  local safe_dataset="${dataset//-/_}"
  local job_base="${JOB_PREFIX}_${safe_model}_${safe_dataset}_${bias}_${probe}"
  [[ "${universal}" == "True" ]] && job_base="${job_base}_universal"

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
echo "Model: ${model} Dataset: ${dataset} Bias: ${bias} Probe: ${probe} Universal: ${universal}"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python main.py \
  --model "${model}" \
  --dataset "${dataset}" \
  --bias "${bias}" \
  --bs_probe ${BS_PROBE} \
  --n_ckpts ${N_CKPTS} \
  --probe "${probe}" \
  --train_probes \
  $([[ "${universal}" == "True" ]] && echo "--universal")
EOF
}

# Submit each specific job combination
for job_spec in "${jobs[@]}"; do
  read -r model dataset bias probe <<< "${job_spec}"
  submit_job "${model}" "${dataset}" "${bias}" "${probe}" "${universal}"
done

