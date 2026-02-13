#!/bin/bash
set -euo pipefail

# General job submission script - mirrors main.py interface
# Usage examples:
#   MODELS=qwen-3-8b DATASETS=mmlu GENERATE=1 ./scripts/submit.sh
#   MODELS=qwen-3-8b,llama-3-1-8b DATASETS=mmlu,arc-challenge BIAS=expert PROBE=mot_vs_alg TRAIN_PROBES=1 EVALUATE_PROBES=1 ./scripts/submit.sh
#   MODELS=all DATASETS=all BIAS=expert PROBE=mot_vs_alg TRAIN_PROBES=1 ./scripts/submit.sh

# Load environment variables from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

# SLURM settings
ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-04:00:00}
MEM=${MEM:-100g}
GPUS=${GPUS:-1}
CPUS=${CPUS:-1}
NODES=${NODES:-1}

# Experiment settings
MODELS=${MODELS:-all}
DATASETS=${DATASETS:-all}
BIAS=${BIAS:-all}
PROBE=${PROBE:-all}
LLM=${LLM:-gpt-5-nano}
HINT_IDX=${HINT_IDX:-0}
TAG=${TAG:-}

# Scale and checkpointing
SCALE=${SCALE:-large}
N_CKPTS=${N_CKPTS:-3}
CKPT=${CKPT:-rel}

# Training options
BALANCED=${BALANCED:-0}
FILTER_MENTIONS=${FILTER_MENTIONS:-1}
UNIVERSAL=${UNIVERSAL:-0}
REASON_FIRST=${REASON_FIRST:-0}

# Actions (set to 1 to enable)
GENERATE=${GENERATE:-0}
EVALUATE=${EVALUATE:-0}
TRAIN_PROBES=${TRAIN_PROBES:-0}
EVALUATE_PROBES=${EVALUATE_PROBES:-0}
EVALUATE_LLM=${EVALUATE_LLM:-0}

# Aggregation options
AGGREGATE_LAYERS=${AGGREGATE_LAYERS:-}
AGGREGATE_STEPS=${AGGREGATE_STEPS:-}

# Predefined sets
ALL_MODELS="qwen-3-8b,llama-3.1-8b,gemma-3-4b"
ALL_DATASETS="mmlu,arc-challenge,commonsense_qa,aqua"
ALL_BIASES="expert,self,metadata"
ALL_PROBES="hint-recovery,mot_vs_alg,mot_vs_res,mot_vs_oth"

# Expand "all" keywords
[[ "${MODELS}" == "all" ]] && MODELS="${ALL_MODELS}"
[[ "${DATASETS}" == "all" ]] && DATASETS="${ALL_DATASETS}"
[[ "${BIAS}" == "all" ]] && BIAS="${ALL_BIASES}"
[[ "${PROBE}" == "all" ]] && PROBE="${ALL_PROBES}"

# Convert comma-separated to arrays
IFS=',' read -ra MODEL_ARR <<< "${MODELS}"
IFS=',' read -ra DATASET_ARR <<< "${DATASETS}"
IFS=',' read -ra BIAS_ARR <<< "${BIAS:-none}"
IFS=',' read -ra PROBE_ARR <<< "${PROBE:-none}"

# Build action flags string (no leading space)
ACTION_FLAGS=""
[[ "${GENERATE}" == "1" ]] && ACTION_FLAGS="${ACTION_FLAGS:+$ACTION_FLAGS }--generate"
[[ "${EVALUATE}" == "1" ]] && ACTION_FLAGS="${ACTION_FLAGS:+$ACTION_FLAGS }--evaluate"
[[ "${TRAIN_PROBES}" == "1" ]] && ACTION_FLAGS="${ACTION_FLAGS:+$ACTION_FLAGS }--train_probes"
[[ "${EVALUATE_PROBES}" == "1" ]] && ACTION_FLAGS="${ACTION_FLAGS:+$ACTION_FLAGS }--evaluate_probes"
[[ "${EVALUATE_LLM}" == "1" ]] && ACTION_FLAGS="${ACTION_FLAGS:+$ACTION_FLAGS }--evaluate_llm"

if [[ -z "${ACTION_FLAGS}" ]]; then
    echo "Error: No action specified. Set at least one of: GENERATE, EVALUATE, TRAIN_PROBES, EVALUATE_PROBES, EVALUATE_LLM"
    exit 1
fi

# Build optional flags (no leading space)
OPTIONAL_FLAGS=""
[[ "${BALANCED}" == "1" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS:+$OPTIONAL_FLAGS }--balanced"
[[ "${UNIVERSAL}" == "1" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS:+$OPTIONAL_FLAGS }--universal"
[[ "${REASON_FIRST}" == "1" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS:+$OPTIONAL_FLAGS }--reason_first"
[[ -n "${TAG}" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS:+$OPTIONAL_FLAGS }--tag ${TAG}"
[[ -n "${AGGREGATE_LAYERS}" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS:+$OPTIONAL_FLAGS }--aggregate_layers ${AGGREGATE_LAYERS}"
[[ -n "${AGGREGATE_STEPS}" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS:+$OPTIONAL_FLAGS }--aggregate_steps ${AGGREGATE_STEPS}"

FILTER_MENTIONS_VAL="True"
[[ "${FILTER_MENTIONS}" == "0" ]] && FILTER_MENTIONS_VAL="False"

# Determine what we're iterating over
needs_bias() {
    [[ "${GENERATE}" == "1" && -n "${BIAS}" && "${BIAS}" != "none" ]] || \
    [[ "${TRAIN_PROBES}" == "1" ]] || \
    [[ "${EVALUATE_PROBES}" == "1" ]] || \
    [[ "${EVALUATE_LLM}" == "1" ]]
}

needs_probe() {
    [[ "${TRAIN_PROBES}" == "1" ]] || \
    [[ "${EVALUATE_PROBES}" == "1" ]] || \
    [[ "${EVALUATE_LLM}" == "1" ]]
}

# Submit jobs
job_count=0
first_job_id=""
for model in "${MODEL_ARR[@]}"; do
    for dataset in "${DATASET_ARR[@]}"; do
        # Determine bias iteration
        if needs_bias; then
            bias_list=("${BIAS_ARR[@]}")
        else
            bias_list=("none")
        fi

        for bias in "${bias_list[@]}"; do
            # Determine probe iteration
            if needs_probe; then
                probe_list=("${PROBE_ARR[@]}")
            else
                probe_list=("none")
            fi

            for probe in "${probe_list[@]}"; do
                # Build job name (use __ as delimiter to avoid splitting issues with commonsense_qa)
                job_name="job__${model}__${dataset}"
                [[ "${bias}" != "none" ]] && job_name="${job_name}__${bias}"
                [[ "${probe}" != "none" ]] && job_name="${job_name}__${probe}"

                # Build bias/probe flags for this job
                job_flags="${ACTION_FLAGS}"
                [[ "${bias}" != "none" ]] && job_flags="${job_flags:+$job_flags }--bias ${bias}"
                [[ "${probe}" != "none" ]] && job_flags="${job_flags:+$job_flags }--probe ${probe}"

                echo "Submitting: ${job_name}"
                echo "  Model: ${model}, Dataset: ${dataset}, Bias: ${bias}, Probe: ${probe}"
                echo "  Actions:${ACTION_FLAGS}"

                job_id=$(sbatch --export=ALL --parsable <<EOF
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

echo "=== Job: ${job_name} ==="
echo "Model: ${model} | Dataset: ${dataset} | Bias: ${bias} | Probe: ${probe}"
echo "Actions:${ACTION_FLAGS}"
echo "Git commit: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

python main.py --model "${model}" --dataset "${dataset}" --scale ${SCALE} --n_ckpts ${N_CKPTS} --ckpt ${CKPT} --hint_idx ${HINT_IDX} --llm ${LLM} --filter_mentions ${FILTER_MENTIONS_VAL} ${job_flags}${OPTIONAL_FLAGS:+ ${OPTIONAL_FLAGS}}

echo "Done!"
EOF
)
                echo "Submitted batch job ${job_id}"
                [ -z "$first_job_id" ] && first_job_id="$job_id"
                job_count=$((job_count + 1))
            done
        done
    done
done

echo ""
echo "Submitted ${job_count} job(s)"

# Save first job ID for monitor.sh
if [ -n "$first_job_id" ]; then
    echo "$first_job_id" > "${SCRIPT_DIR}/.last_submit"
    echo "Monitor with: ./scripts/monitor.sh"
fi
