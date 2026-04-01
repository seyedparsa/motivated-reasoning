#!/bin/bash
set -euo pipefail

# Submit cross-dataset and cross-hint transfer experiments
# Usage:
#   ./scripts/submit_cross_eval.sh              # submit all
#   MODE=cross_dataset ./scripts/submit_cross_eval.sh   # only cross-dataset
#   MODE=cross_hint ./scripts/submit_cross_eval.sh      # only cross-hint
#   MODE=permuted ./scripts/submit_cross_eval.sh        # only permutation baselines

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.env"

ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT}}
PARTITION=${PARTITION:-${SLURM_PARTITION}}
TIME=${TIME:-01:00:00}
MEM=${MEM:-100g}
GPUS=${GPUS:-1}

MODE=${MODE:-all}
SCALE=${SCALE:-small}
PROBE=${PROBE:-mot_vs_alg}
N_CKPTS=${N_CKPTS:-3}
CKPT=${CKPT:-rel}

ALL_MODELS="qwen-3-8b,llama-3.1-8b,gemma-3-4b"
ALL_DATASETS="mmlu,arc-challenge,commonsense_qa,aqua"
ALL_BIASES="expert,self,metadata"

IFS=',' read -ra MODELS <<< "${ALL_MODELS}"
IFS=',' read -ra DATASETS <<< "${ALL_DATASETS}"
IFS=',' read -ra BIASES <<< "${ALL_BIASES}"

job_count=0

submit_job() {
    local job_name=$1
    local cmd=$2

    job_id=$(sbatch --parsable --account="${ACCOUNT}" --partition="${PARTITION}" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="${MEM}" \
        --gpus-per-node="${GPUS}" --time="${TIME}" \
        --job-name="${job_name}" \
        --output="${job_name}_%j.out" --error="${job_name}_%j.err" \
        <<SLURM_EOF
#!/bin/bash
set -euo pipefail
cd ~/neural_controllers
source .env
source ~/.bashrc
conda activate \${CONDA_ENV}
export HF_HOME MOTIVATION_HOME HF_TOKEN HF_USE_SOFTFILELOCK=1 PYTHONUNBUFFERED=1

echo "=== ${job_name} ==="
${cmd}
echo "Done!"
SLURM_EOF
    )
    echo "  Submitted ${job_name} -> ${job_id}"
    job_count=$((job_count + 1))
}

# Cross-dataset transfer: train on A, eval on B (same model, same bias)
if [[ "${MODE}" == "all" || "${MODE}" == "cross_dataset" ]]; then
    echo "=== Cross-dataset transfer ==="
    for model in "${MODELS[@]}"; do
        for bias in "${BIASES[@]}"; do
            for train_ds in "${DATASETS[@]}"; do
                for eval_ds in "${DATASETS[@]}"; do
                    [[ "${train_ds}" == "${eval_ds}" ]] && continue
                    job_name="xd__${model}__${train_ds}__${eval_ds}__${bias}"
                    cmd="python main.py --model ${model} --dataset ${train_ds} --bias ${bias} --probe ${PROBE} --evaluate_probes --eval_dataset ${eval_ds} --n_ckpts ${N_CKPTS} --ckpt ${CKPT} --scale ${SCALE}"
                    submit_job "${job_name}" "${cmd}"
                done
            done
        done
    done
fi

# Cross-hint transfer: train on bias A, eval on bias B (same model, same dataset)
if [[ "${MODE}" == "all" || "${MODE}" == "cross_hint" ]]; then
    echo "=== Cross-hint transfer ==="
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for train_bias in "${BIASES[@]}"; do
                for eval_bias in "${BIASES[@]}"; do
                    [[ "${train_bias}" == "${eval_bias}" ]] && continue
                    job_name="xb__${model}__${dataset}__${train_bias}__${eval_bias}"
                    cmd="python main.py --model ${model} --dataset ${dataset} --bias ${train_bias} --probe ${PROBE} --evaluate_probes --eval_bias ${eval_bias} --n_ckpts ${N_CKPTS} --ckpt ${CKPT} --scale ${SCALE}"
                    submit_job "${job_name}" "${cmd}"
                done
            done
        done
    done
fi

# Permutation baselines: a few representative configs
if [[ "${MODE}" == "all" || "${MODE}" == "permuted" ]]; then
    echo "=== Permutation baselines ==="
    for model in "${MODELS[@]}"; do
        for bias in "${BIASES[@]}"; do
            # Standard (train=eval) with permuted labels
            job_name="perm__${model}__mmlu__${bias}"
            cmd="python main.py --model ${model} --dataset mmlu --bias ${bias} --probe ${PROBE} --evaluate_probes --n_ckpts ${N_CKPTS} --ckpt ${CKPT} --scale ${SCALE} --permute_eval_labels"
            submit_job "${job_name}" "${cmd}"
        done
    done
fi

# Cross-model transfer: train on model A, eval on model B (same dataset, same bias)
if [[ "${MODE}" == "all" || "${MODE}" == "cross_model" ]]; then
    echo "=== Cross-model transfer ==="
    for train_model in "${MODELS[@]}"; do
        for eval_model in "${MODELS[@]}"; do
            [[ "${train_model}" == "${eval_model}" ]] && continue
            for dataset in "${DATASETS[@]}"; do
                for bias in "${BIASES[@]}"; do
                    job_name="xm__${train_model}__${eval_model}__${dataset}__${bias}"
                    cmd="python main.py --model ${train_model} --dataset ${dataset} --bias ${bias} --probe ${PROBE} --evaluate_probes --eval_model ${eval_model} --n_ckpts ${N_CKPTS} --ckpt ${CKPT} --scale ${SCALE}"
                    submit_job "${job_name}" "${cmd}"
                done
            done
        done
    done
fi

echo ""
echo "Submitted ${job_count} job(s)"
