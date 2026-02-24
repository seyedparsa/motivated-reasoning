# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research framework for detecting **motivated reasoning** in language models (see `main.pdf` for the paper). The core insight is that LLMs can produce unfaithful chain-of-thought (CoT) reasoning: when given a hint pointing to a specific answer, the model may switch its answer to match the hint while rationalizing that choice without acknowledging the hint's influence.

This project uses supervised probes on residual-stream activations to detect motivated reasoning even when CoT monitoring fails to reveal it.

## Research Framework

**Paired Context Evaluation**: For each question, compare model behavior on:
- Unhinted prompt x⊥(q): question only
- Hinted prompt xₕ(q): question + hint suggesting answer h

**Transition Categories** (how the model reacts to hints):
| Category | Condition | Meaning |
|----------|-----------|---------|
| Motivated | a⊥ ≠ h, aₕ = h | Switched answer to match hint |
| Resistant | a⊥ ≠ h, aₕ = a⊥ | Ignored hint, kept original answer |
| Aligned | a⊥ = h, aₕ = h | Would have chosen hinted answer anyway |
| Departing | a⊥ = h, aₕ ≠ h | Moved away from hinted choice |
| Shifting | a⊥ ≠ h, aₕ ≠ a⊥, aₕ ≠ h | Changed to different non-hinted choice |

**Three Detection Tasks**:
1. **Hint Recovery**: Can probes recover which answer was hinted from internal representations at end of CoT?
2. **Post-hoc Detection**: Distinguish *motivated* from *aligned* cases (both end with hinted answer, but only motivated was influenced by hint)
3. **Preemptive Detection**: Predict *before* CoT generation whether model will follow or resist the hint

## Build & Development Commands

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Generate responses (with Chain-of-Thought reasoning)
python main.py --generate --model qwen-3-8b --dataset arc-challenge --reason_first

# Generate with biased context
python main.py --generate --model qwen-3-8b --dataset arc-challenge --reason_first \
    --bias expert --hint_idx 0

# Evaluate generated responses
python main.py --evaluate --model qwen-3-8b --dataset arc-challenge

# Train probes
python main.py --train_probes --model qwen-3-8b --dataset arc-challenge \
    --probe has-switched --bias expert --n_ckpts 3 --ckpt rel

# Evaluate probes
python main.py --evaluate_probes --model qwen-3-8b --dataset arc-challenge \
    --probe has-switched --n_test_questions 200

# Quick smoke test (use before scaling up)
python main.py --generate --model qwen-3-8b --dataset arc-challenge \
    --n_questions 40 --split test
```

## Job Submission & Monitoring (SLURM)

**General submission**: Use `scripts/submit.sh` for all job types. It mirrors the `main.py` interface and iterates over models/datasets/biases/probes.

```bash
# Submit LLM evaluation jobs
MODELS=qwen-3-8b,llama-3.1-8b,gemma-3-4b DATASETS=aqua BIAS=self PROBE=mot_vs_oth EVALUATE_LLM=1 bash scripts/submit.sh

# Submit probe training + evaluation
MODELS=all DATASETS=all BIAS=expert PROBE=mot_vs_alg TRAIN_PROBES=1 EVALUATE_PROBES=1 bash scripts/submit.sh

# Submit generation
MODELS=qwen-3-8b DATASETS=mmlu GENERATE=1 REASON_FIRST=1 bash scripts/submit.sh

# Override SLURM defaults
TIME=16:00:00 MEM=200g GPUS=2 MODELS=qwen-3-8b DATASETS=mmlu TRAIN_PROBES=1 bash scripts/submit.sh
```

**Key environment variables for `submit.sh`**:
- `MODELS`, `DATASETS`, `BIAS`, `PROBE`: Comma-separated lists or `all`
- Actions: `GENERATE=1`, `EVALUATE=1`, `TRAIN_PROBES=1`, `EVALUATE_PROBES=1`, `EVALUATE_LLM=1`
- SLURM: `TIME` (default 08:00:00), `MEM` (100g), `GPUS` (1), `EXCLUDE` (bad nodes)
- Options: `BALANCED`, `FILTER_MENTIONS`, `UNIVERSAL`, `REASON_FIRST`, `SCALE`, `N_CKPTS`, `CKPT`, `TAG`, `LLM`, `HINT_IDX`
- `NO_UPDATE_LAST=1`: Don't update `.last_submit` tracker

**Monitoring**: Use `scripts/monitor.sh` to check job status. It reads the first job ID from `scripts/.last_submit` and reports running, completed, failed, and stuck jobs.

```bash
bash scripts/monitor.sh
```

## Results Databases

Results are stored in SQLite databases at `/work/hdd/bbjr/pmirtaheri/motivated/`:
- `probe_metrics.db` — Probe training/evaluation results (RFM and linear accuracy/AUC per model/dataset/bias/probe/layer/step)
- `llm_metrics.db` — LLM baseline evaluation results

Use `TAG` in `submit.sh` to differentiate experimental runs (tag is part of the primary key).

## Architecture

**Entry Point**: `main.py` - CLI for generation, evaluation, and probe training

**Core Package**: `thoughts/`
- `multiple_choice.py` - Main workflow functions (~1600 lines): `generate_responses()`, `evaluate_responses()`, `train_probes()`, `evaluate_probes()`
- `utils.py` - Model/dataset/tokenizer loading via `get_model()`, `get_dataset()`, `get_tokenizer()`
- `configs/models.json` - Supported models: Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-3-4B
- `configs/datasets.json` - Supported datasets: MMLU, ARC-Challenge, CommonsenseQA, AQuA

**Root Modules**:
- `neural_controllers.py` - `NeuralController` class for steering and detection
- `control_toolkits.py` - Toolkit classes (RFM, Linear, Logistic, PCA, MeanDifference)
- `direction_utils.py` - Hidden state extraction and probe training

**Data Flow**:
1. Load model/dataset from HuggingFace
2. Prepare prompts with optional bias injection
3. Generate responses, extract answers via regex
4. Train probes on hidden states at multiple layers/checkpoints
5. Output artifacts to `outputs/` and upload to HuggingFace Hub

## Key Patterns

**Hint/Bias Types** (passed via `--bias`):
| Flag | Paper Name | Prompt Format |
|------|------------|---------------|
| `expert` | Sycophancy | "An expert indicates the answer is {X}" |
| `self` | Consistency | Prefilled assistant response "The answer is {X}" |
| `metadata` | Metadata | XML tags `<correct-choice>{X}</correct-choice>` |

**Probe Types** (in `control_toolkits.py`):
- **RFM** (primary): Recursive Feature Machines - learns non-linear feature maps via AGOP
- **Linear**: Ridge regression on hidden states
- **Logistic**: Sklearn logistic regression
- **MeanDifference**: Simple class-wise mean difference (binary only)
- **PCA**: Principal component analysis on paired examples (binary only)

**Layer Indexing**: Negative indices (`-1` = final layer, `-2` = second-to-last)

**NeuralController Usage**:
```python
controller = NeuralController(model, tokenizer, control_method='rfm', n_components=5)
controller.compute_directions(train_data, train_labels)
controller.generate(prompt, layers_to_control=list(range(-1, -11, -1)), control_coef=0.5)
```

**Hidden States**: `Dict[int, torch.Tensor]` where keys are negative layer indices

**Probe Checkpointing**: `--ckpt rel` (relative positions), `prefix`, or `suffix`

## File Naming Convention

Output files follow: `{model}_{dataset}_{split}_{bias}_{task}.{ext}`

This enables automatic globbing in downstream analysis notebooks.

## Testing

No pytest suite. Use deterministic slices for smoke tests:
- `--n_questions 40 --split train` for quick validation
- Always pair `--generate` with `--evaluate` to catch parsing regressions
- Compare new CSVs in `outputs/probe_metrics/` against `old_outputs/` baselines

## Environment

- API keys (`OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`) go in shell profile or `simons*.yml`
- `MOTIVATION_HOME` defaults to `outputs/`
- SLURM scripts in `scripts/` for HPC submission

## Directories

- Probe checkpoints have been moved from `/work/hdd/bbjr/pmirtaheri/motivated/probes/` to `/var/tmp/pmirtaheri/probes/` on `gh-login03` (node-local NVMe, not accessible from compute nodes)
- `outputs/` - Generated responses, probe metrics, checkpoints
- `figures/` - Publication figures (bias_detection/, taxonomy/)
- `notebooks/` - Jupyter notebooks for analysis
- `analysis/` - Post-hoc analysis and plotting scripts
- `old_outputs/` - Archive for regression comparison
- `core/` - 1.2 GB crash dump (avoid editing)

## Paper Reference

See `main.pdf` for the full paper: "Detecting Motivated Reasoning in Internal Representations of Language Models" (under review at ICML).
