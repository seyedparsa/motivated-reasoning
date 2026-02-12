#!/bin/bash
set -euo pipefail

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

MODEL=${MODEL:-qwen-2.5-7b}
DATASET=${DATASET:-mmlu}
SPLIT=${SPLIT:-test}
BATCH_SIZE=${BATCH_SIZE:-64}
HINTS="${HINTS:-0 1 2 3 4}"          # space-separated hint indices
ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-02:00:00}
MEM=${MEM:-100g}
GPUS=${GPUS:-1}
CPUS=${CPUS:-1}
NODES=${NODES:-1}
JOB_PREFIX=${JOB_PREFIX:-generate}
DEBUG=${DEBUG:-0}
ACTIVATE=${ACTIVATE:-}         # e.g. "source ~/.bashrc && conda activate env"

echo "Submitting jobs model=${MODEL} dataset=${DATASET} split=${SPLIT} hints=[${HINTS}]"

if [[ -n "${ACTIVATE}" ]]; then
  echo "(Activation command will run inside each job)"
fi

dbg_line='[[ "${DEBUG}" == "1" ]] && set -x'

submit_job() {
  local label=$1     # suffix label
  local bias=$2      # expert | self | metadata | ''
  local hint=$3      # '' or integer
  local rf_flag=$4   # --reason_first or ''
  # Determine condition label: expert/self for biased runs; otherwise rf/af for baselines
  local cond
  if [[ -n ${bias} ]]; then
    cond=${bias}
  else
    if [[ -n ${rf_flag} ]]; then
      cond="rf"
    else
      cond="af"
    fi
  fi
  local hint_suffix=""
  if [[ -n ${hint} ]]; then
    hint_suffix="${hint}"
  fi
  local job_name="${MODEL}_${DATASET}_${cond}${hint_suffix}_bs${BATCH_SIZE}"
  local bias_arg=""; [[ -n ${bias} ]] && bias_arg="--bias ${bias}"
  local hint_arg=""; [[ -n ${hint} ]] && hint_arg="--hint_idx ${hint}"
  sbatch <<EOF
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
[[ "${DEBUG}" == "1" ]] && set -x
[[ -n "${ACTIVATE}" ]] && eval "${ACTIVATE}"
echo "Job: ${job_name} bias=${bias:-none} hint=${hint:-none} reason_first=${rf_flag:+yes}"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python main.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --generate \
  --bs_gen ${BATCH_SIZE} \
  --n_gen 100 \
  ${rf_flag} \
  ${bias_arg} \
  ${hint_arg}
EOF
}

for H in ${HINTS}; do
  submit_job "meta_hint${H}" metadata ${H} --reason_first
  submit_job "expe_hint${H}" expert ${H} --reason_first
  submit_job "self_hint${H}" self   ${H} --reason_first
done

submit_job baseline_rf '' '' --reason_first
submit_job baseline_af '' '' ''

# total_hints=$(echo ${HINTS} | wc -w)
# echo "Submitted $(( total_hints * 2 )) hinted + 2 baseline = $(( total_hints * 2 + 2 )) jobs." 
