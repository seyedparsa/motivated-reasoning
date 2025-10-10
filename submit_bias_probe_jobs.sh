#!/bin/bash
set -euo pipefail

# Submit bias-probe evaluation jobs across models, datasets, and bias types.

ACCOUNT=${ACCOUNT:-bbjr-dtai-gh}
PARTITION=${PARTITION:-ghx4}
TIME=${TIME:-04:00:00}
MEM=${MEM:-64g}
GPUS=${GPUS:-1}
CPUS=${CPUS:-1}
NODES=${NODES:-1}
JOB_PREFIX=${JOB_PREFIX:-bias_probe}
BS_PROBE=${BS_PROBE:-32}
N_CKPTS=${N_CKPTS:-5}
N_TRAIN=${N_TRAIN:-5000}
ACTIVATE=${ACTIVATE:-}
DEBUG=${DEBUG:-0}

models=(qwen-3-8b llama-3.1-8b gemma-3-4b)
datasets=(mmlu aqua commonsense_qa arc-challenge)
biases=(self expert metadata)

submit_job() {
  local model="$1"
  local dataset="$2"
  local bias="$3"

  local safe_model="${model//-/_}"
  safe_model="${safe_model//./_}"
  local safe_dataset="${dataset//-/_}"
  local job_base="${JOB_PREFIX}_${safe_model}_${safe_dataset}_${bias}"

  echo "Submitting ${job_base}"

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
#SBATCH --job-name=${job_base}
#SBATCH --output=${job_base}_%j.out
#SBATCH --error=${job_base}_%j.err

set -euo pipefail
[[ "${DEBUG}" == "1" ]] && set -x
[[ -n "${ACTIVATE}" ]] && eval "${ACTIVATE}"

echo "Job: ${job_base}"
echo "Model: ${model} Dataset: ${dataset} Bias: ${bias}"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python thoughts_eval.py \
  --model "${model}" \
  --dataset "${dataset}" \
  --probe bias \
  --bias "${bias}" \
  --bs_probe ${BS_PROBE} \
  --n_ckpts ${N_CKPTS} \
  --n_train ${N_TRAIN}
EOF
}

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for bias in "${biases[@]}"; do
      submit_job "${model}" "${dataset}" "${bias}"
    done
  done
done
