#!/bin/bash
set -euo pipefail

# Submit probe training job
# Usage: MODEL=qwen-3-8b DATASET=mmlu BIAS=expert PROBE=mot_vs_alg ./scripts/submit_probes.sh

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
N_CKPTS=${N_CKPTS:-3}
CKPT=${CKPT:-rel}
SCALE=${SCALE:-small}

# Default configuration
MODEL=${MODEL:-qwen-3-8b}
DATASET=${DATASET:-mmlu}
BIAS=${BIAS:-expert}
PROBE=${PROBE:-mot_vs_alg}

# Training options (set to 1 to enable, 0 to disable)
BALANCED=${BALANCED:-0}
FILTER_MENTIONS=${FILTER_MENTIONS:-1}
UNIVERSAL=${UNIVERSAL:-0}

# Build optional flags
OPTIONAL_FLAGS=""
[[ "${BALANCED}" == "1" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --balanced"
[[ "${UNIVERSAL}" == "1" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --universal"
FILTER_MENTIONS_VAL="True"
[[ "${FILTER_MENTIONS}" == "0" ]] && FILTER_MENTIONS_VAL="False"

job_name="train_${PROBE}_${MODEL}_${DATASET}_${BIAS}"

echo "Submitting train probe job: ${job_name}"
echo "Model: ${MODEL}, Dataset: ${DATASET}, Bias: ${BIAS}, Probe: ${PROBE}"
echo "Options: balanced=${BALANCED}, filter_mentions=${FILTER_MENTIONS}, universal=${UNIVERSAL}"

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

echo "=== Probe Training ==="
echo "Job: ${job_name}"
echo "Model: ${MODEL} Dataset: ${DATASET} Bias: ${BIAS} Probe: ${PROBE}"
echo "Options: balanced=${BALANCED}, filter_mentions=${FILTER_MENTIONS}, universal=${UNIVERSAL}"
echo "Git commit: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

echo "Training ${PROBE} probes..."
python main.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --bias "${BIAS}" \
  --probe "${PROBE}" \
  --train_probes \
  --n_ckpts ${N_CKPTS} \
  --ckpt ${CKPT} \
  --scale ${SCALE} \
  --filter_mentions ${FILTER_MENTIONS_VAL} \
  ${OPTIONAL_FLAGS}

echo "Done!"
EOF
