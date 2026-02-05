# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the CLI entry point for generation, evaluation, and probe training. The `thoughts/` package houses dataset loaders, prompt builders, and probe utilities (`multiple_choice.py`, `utils.py`, configs, and cached data). Helper modules such as `generation_utils.py`, `control_toolkits.py`, and `direction_utils.py` sit at the repo root for cross-cutting logic. Experimental notebooks live under `notebooks/`, publication assets under `figures/`, and large outputs or zipped probe logs are kept inside `outputs/` and `old_outputs/`. Avoid editing the 1.2 GB crash dump in `core`.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs the Hugging Face, OpenAI, and Torch stack expected by the scripts.
- `python main.py --generate --model qwen-3-8b --dataset arc-challenge --bias expert --hint_idx 0` runs the full reasoning/generation pass and writes artifacts to `outputs/`.
- `python main.py --evaluate --model qwen-3-8b --dataset arc-challenge` scores the most recent generations for the specified split.
- `python main.py --train_probes --probe has-switched --n_ckpts 3 --ckpt rel` fits probes and saves checkpoints under `outputs/probe_metrics/`.
- `python main.py --evaluate_probes ... --n_test_questions 200` or `bash run_thoughts_eval.sh` reproduces the HPC evaluation template; log files land next to the script.

## Coding Style & Naming Conventions
Follow PEP 8: four-space indentation, snake_case functions (for example `extract_questions`), and explicit arguments instead of positional magic. Keep helpers pure where possible and pass dataset/model names through to support grid runs. Type hints are encouraged for new modules, and every public function should include a concise docstring describing expected tensors or datasets. Place reusable prompt templates in `thoughts/configs/` rather than scattering literals across scripts.

## Testing Guidelines
There is no pytest suite yet, so rely on deterministic slices. Use `--n_questions 40 --split train` to produce quick smoke runs before scaling up. Always pair `--generate` with an immediate `--evaluate` to catch parsing regressions. For probes, compare the new CSVs in `outputs/probe_metrics/` against the baselines in `old_outputs/` and flag any drop larger than 1 pp. Name new result files with the `{model}_{dataset}_{split}_{bias}_{task}` pattern so downstream notebooks can glob them automatically.

## Commit & Pull Request Guidelines
Git history favors short, imperative subjects (`modify load_data`, `clean`). Follow that format, keep subjects under 50 characters, and explain rationale plus reproduction commands in the body. Each PR should: (1) describe the scenario solved, (2) link the dataset or issue ID, (3) paste the exact CLI invocations and key metrics from `outputs/`, and (4) attach any plot previews from `figures/` if visuals changed. Mention whether regenerated artifacts were uploaded or intentionally omitted.

## Security & Configuration Tips
Store API keys (for example `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`) in your shell profile or the provided `simons*.yml` environment files, never in code. When adapting the SLURM scripts, keep allocation lines intact and prefer overriding parameters via environment variables rather than editing secrets into the files. Large dataset dumps belong in versioned archives under `outputs/`; commit only metadata or sample rows, and document download steps inside `FUTURE_IMPROVEMENTS.md` if new resources are required.

