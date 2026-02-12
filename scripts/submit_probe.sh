#!/bin/bash
set -euo pipefail

# Submit probe training job

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-02:00:00}
MEM=${MEM:-100g}
GPUS=${GPUS:-1}
CPUS=${CPUS:-1}
NODES=${NODES:-1}
BS_PROBE=${BS_PROBE:-32}
N_CKPTS=${N_CKPTS:-3}

# Default test configuration
MODEL=${MODEL:-qwen-3-8b}
DATASET=${DATASET:-arc-challenge}
BIAS=${BIAS:-expert}
PROBE=${PROBE:-bias}

job_name="probe_${PROBE}_${MODEL}_${DATASET}_${BIAS}"

echo "Submitting probe job: ${job_name}"
echo "Model: ${MODEL}, Dataset: ${DATASET}, Bias: ${BIAS}, Probe: ${PROBE}"

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
source \${CONDA_SH}
conda activate \${CONDA_ENV}

# Set environment variables
export HF_HOME
export MOTIVATION_HOME
export HF_TOKEN
export HF_USE_SOFTFILELOCK=1
export PYTHONUNBUFFERED=1

echo "=== Probe Training Experiment ==="
echo "Job: ${job_name}"
echo "Model: ${MODEL} Dataset: ${DATASET} Bias: ${BIAS} Probe: ${PROBE}"
echo "Git commit: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

echo "Training ${PROBE} probes..."
python main.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --bias "${BIAS}" \
  --probe "${PROBE}" \
  --train_probes \
  --bs_probe ${BS_PROBE} \
  --n_ckpts ${N_CKPTS} \
  --balanced

echo ""
echo "Evaluating ${PROBE} probes..."
python main.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --bias "${BIAS}" \
  --probe "${PROBE}" \
  --evaluate_probes \
  --bs_probe ${BS_PROBE} \
  --n_ckpts ${N_CKPTS} \
  --balanced

echo "Done!"
EOF
