# Catching Rationalization in the Act

Code for [*Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing*](https://arxiv.org/abs/2603.17199) by Parsa Mirtaheri and Mikhail Belkin (UC San Diego).

## Abstract

Large language models (LLMs) can produce chains of thought (CoT) that do not accurately reflect the actual factors driving their answers. In multiple-choice settings with an injected hint favoring a particular option, models may shift their final answer toward the hinted option and produce a CoT that rationalizes the response without acknowledging the hint -- an instance of motivated reasoning. Using supervised probes trained on the model's residual stream, we show that (i) **pre-generation** probes, applied before any CoT tokens are generated, predict motivated reasoning as well as an LLM-based CoT monitor that accesses the full CoT trace, and (ii) **post-generation** probes, applied after CoT generation, outperform the same monitor.

<p align="center">
  <img src="figures/pre_generation_probe_vs_cot_monitor.png" width="700">
</p>

**Figure 1.** Pre-generation probe AUC (y-axis) vs. GPT-5-nano CoT monitor AUC (x-axis) for motivated reasoning detection. Each point is a hint type averaged across datasets. The probe operates on hidden states *before* CoT generation; the CoT monitor reads the *full* reasoning trace. Points above the diagonal indicate the probe is better.

## Problem Setup

We adopt a **paired context evaluation** framework. For each multiple-choice question *q*, we construct:
- An **unhinted prompt** containing only the question
- A **hinted prompt** containing the question plus a hint suggesting answer *h*

The model generates a CoT and final answer for each prompt. By comparing the final answers, we categorize each response:

| Category | Condition | Meaning |
|---|---|---|
| **Motivated** | a⊥ ≠ h, aₕ = h | Model switches its answer to match the hint |
| **Resistant** | a⊥ ≠ h, aₕ = a⊥ | Model ignores the hint, preserves its original answer |
| **Aligned** | a⊥ = h, aₕ = h | Model would have chosen the hinted answer anyway |

We use three hint types:

| Hint Type | Description | Format |
|---|---|---|
| Sycophancy | Expert suggestion | "An expert indicates the answer is {X}" |
| Consistency | Prefilled model response | Assistant: "The answer is {X}" |
| Metadata | Embedded in XML tags | `<correct-choice>{X}</correct-choice>` |

## Detection Tasks

1. **Pre-generation motivated reasoning detection.** Can a probe predict, *before any CoT tokens are generated*, whether the model will be motivated by the hint? Our pre-generation probes achieve performance comparable to a GPT-5-nano CoT monitor that has access to the full reasoning trace.

2. **Post-generation motivated reasoning detection.** Given the model's internal representations at the end of CoT, can a probe distinguish *motivated* from *aligned* cases? Our post-generation probes outperform the GPT-5-nano CoT monitor.

3. **Hint recovery.** Can a probe recover the hinted choice from internal representations along the CoT? Hint-recovery accuracy follows a U-shaped pattern across CoT tokens -- high at the beginning, near chance in the middle, and rising again near the end -- suggesting the model re-engages with the hint as it approaches its final answer, even when the CoT never mentions the hint.

## Models and Benchmarks

**Models:** Qwen-3-8B, Llama-3.1-8B-Instruct, Gemma-3-4B-IT

**Benchmarks:** MMLU, ARC-Challenge, CommonsenseQA, AQuA

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Generation

```bash
# Generate CoT responses
python main.py --generate --model qwen-3-8b --dataset arc-challenge --reason_first

# Generate with a hint (sycophancy, pointing to the first choice)
python main.py --generate --model qwen-3-8b --dataset arc-challenge --reason_first \
    --bias expert --hint_idx 0
```

### Evaluation

```bash
python main.py --evaluate --model qwen-3-8b --dataset arc-challenge
```

### Probe Training and Evaluation

```bash
# Train probes (post-generation: motivated vs aligned)
python main.py --train_probes --model qwen-3-8b --dataset arc-challenge \
    --probe mot_vs_alg --bias expert --n_ckpts 3 --ckpt rel

# Evaluate probes
python main.py --evaluate_probes --model qwen-3-8b --dataset arc-challenge \
    --probe mot_vs_alg --n_test_questions 200
```

Available probe tasks: `h_recovery` (hint recovery), `mot_vs_alg` (motivated vs aligned), `mot_vs_res` (motivated vs resistant), `mot_vs_oth` (motivated vs all others).

### Interactive Mode

```bash
python main.py --interactive --model qwen-3-8b --probe mot_vs_oth
```

### SLURM Job Submission

```bash
# Submit probe training + evaluation across all configs
MODELS=all DATASETS=all BIAS=expert PROBE=mot_vs_alg TRAIN_PROBES=1 EVALUATE_PROBES=1 bash scripts/submit.sh

# Monitor jobs
bash scripts/monitor.sh
```

## Repository Structure

```
main.py                     # CLI entry point
core/
  motivated_reasoning.py    # Main workflow: generation, evaluation, probe training
  probes.py                 # RFM and linear probe training and evaluation
  utils.py                  # Model/dataset/tokenizer loading
  results_db.py             # SQLite persistence for metrics
  configs/
    models.json             # Model registry
    datasets.json           # Dataset registry
analysis/                   # Plotting scripts for paper figures
scripts/
  submit.sh                 # SLURM job submission
  monitor.sh                # Job status monitoring
```

## Citation

```bibtex
@article{mirtaheri2025catching,
  title={Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing},
  author={Mirtaheri, Parsa and Belkin, Mikhail},
  journal={arXiv preprint arXiv:2603.17199},
  year={2025}
}
```
