#!/usr/bin/env python3
"""
Compute bias-detection probe performance across bias types and models.

Reads the CSV logs in outputs/probe_metrics and produces a bar chart similar
to "Bias Detection — Bias Types", where each bar represents the best AUC
achieved by a model for detecting a given bias type averaged over datasets.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Set larger font sizes
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
})

MOTIVATION_HOME = Path(os.environ.get("MOTIVATION_HOME", "/work/hdd/bbjr/pmirtaheri/motivated"))

METRICS_DIR = Path("outputs/probe_metrics")
FIGURES_DIR = MOTIVATION_HOME / "figures"

BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}
BIAS_ORDER = ["Sycophancy", "Consistency", "Metadata"]

MODEL_LABELS = {
    "qwen-3-8b": "Qwen-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gemma-3-4b": "Gemma-3-4B",
}
MODEL_ORDER = ["qwen-3-8b", "llama-3.1-8b", "gemma-3-4b"]
MODEL_COLORS = {
    "qwen-3-8b": "#8856a7",      # Qwen purple
    "llama-3.1-8b": "#e34a33",   # Llama red
    "gemma-3-4b": "#f4c20d",     # Gemma yellow
}


def load_best_auc_per_model(metrics_dir: Path, task: str = "bias", step_mode: str = "last",
                            use_best_layer: bool = False, fixed_best_layer: bool = False,
                            metric: str = "auc") -> pd.DataFrame:
    """Load best metric per model/dataset/bias from SQLite DB, falling back to CSVs.

    step_mode options:
      - "best": best AUC across all steps and layers
      - "first": best AUC from the first step only
      - "last": best AUC from the last step only

    use_best_layer: If True, use best layer per (model, dataset, bias).
    fixed_best_layer: If True, find a single best layer per model (averaged
        across all datasets/biases) and use that layer for all results.
    metric: "auc" or "accuracy".
    """
    # Try SQLite first
    db_path = MOTIVATION_HOME / "probe_metrics.db"
    if db_path.exists():
        return _load_best_auc_from_db(db_path, task, step_mode, use_best_layer, metric,
                                      fixed_best_layer=fixed_best_layer)
    return _load_best_auc_from_csvs(metrics_dir, task, step_mode)


def _load_best_auc_from_db(db_path: Path, task: str, step_mode: str, use_best_layer: bool = False,
                           metric: str = "auc", fixed_best_layer: bool = False) -> pd.DataFrame:
    """Load best metric per model/dataset/bias from the SQLite database."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT model, dataset, bias, layer, step, auc, accuracy FROM probe_metrics WHERE probe = ? AND classifier = 'rfm' AND ckpt_mode = 'rel' AND n_ckpts = 3",
        conn,
        params=(task,),
    )
    conn.close()
    if df.empty:
        raise FileNotFoundError(f"No probe metrics for task={task} in {db_path}")
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df = df.dropna(subset=["layer"])
    # Convert accuracy from percentage to fraction
    if "accuracy" in df.columns:
        df["accuracy"] = df["accuracy"] / 100.0

    # Pre-filter to the relevant step
    step_filtered = []
    for (model, dataset, bias), group in df.groupby(["model", "dataset", "bias"]):
        if step_mode == "first":
            step_filtered.append(group[group["step"] == group["step"].min()])
        elif step_mode == "last":
            step_filtered.append(group[group["step"] == group["step"].max()])
        else:
            step_filtered.append(group)
    df_step = pd.concat(step_filtered)

    # Find the single best layer per model if requested
    if fixed_best_layer:
        best_layer_per_model = (
            df_step.groupby(["model", "layer"])[metric].mean()
            .reset_index()
            .loc[lambda x: x.groupby("model")[metric].idxmax()]
            .set_index("model")["layer"]
            .to_dict()
        )
        print(f"Fixed best layers: {best_layer_per_model}")

    records = []
    for (model, dataset, bias), group in df_step.groupby(["model", "dataset", "bias"]):
        if fixed_best_layer:
            layer = best_layer_per_model[model]
            rows = group[group["layer"] == layer]
            if rows.empty:
                continue
            row = rows.iloc[0]
        elif use_best_layer:
            row = group.loc[group[metric].idxmax()]
        else:
            max_layer = group["layer"].max()
            row = group[group["layer"] == max_layer].iloc[0]
        records.append({"model": row["model"], "dataset": row["dataset"], "bias": row["bias"], "value": row[metric]})
    combined = pd.DataFrame(records)
    combined["bias_label"] = combined["bias"].map(BIAS_LABELS)
    combined = combined.dropna(subset=["bias_label"])
    return combined


def _load_best_auc_from_csvs(metrics_dir: Path, task: str, step_mode: str) -> pd.DataFrame:
    """Fallback: load best AUC from per-run CSV files."""
    records: List[pd.DataFrame] = []
    for path in metrics_dir.glob(f"probe_metrics_*_{task}_per-step_3rel.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        # Filter to RFM classifier if column exists
        if "classifier" in df.columns:
            df = df[df["classifier"] == "rfm"]
        auc_col = "auc" if "auc" in df.columns else "rfm_auc"
        if step_mode == "first":
            step_value = df["step"].min()
            step_df = df[df["step"] == step_value]
        elif step_mode == "last":
            step_value = df["step"].max()
            step_df = df[df["step"] == step_value]
        else:  # "best" - use all steps
            step_df = df
        if step_df.empty:
            continue
        best_row = step_df.loc[step_df[auc_col].idxmax(), ["model", "dataset", "bias", auc_col]]
        if auc_col != "auc":
            best_row = best_row.rename({auc_col: "auc"})
        records.append(best_row.to_frame().T)
    if not records:
        raise FileNotFoundError(f"No bias probe metrics found in {metrics_dir}")
    combined = pd.concat(records, ignore_index=True)
    combined["bias_label"] = combined["bias"].map(BIAS_LABELS)
    combined = combined.dropna(subset=["bias_label"])
    return combined


def aggregate_by_bias(best_auc_df: pd.DataFrame) -> pd.DataFrame:
    """Average best metric over datasets for each model/bias."""
    per_model_bias = (
        best_auc_df.groupby(["model", "bias_label"])["value"].mean().reset_index()
    )
    return per_model_bias


def save_plot(fig, base_name: str, formats: List[str], output_dir: Path = None) -> None:
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_dir / f"{base_name}.{fmt}"
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")


def plot_bias_detection(per_model_bias: pd.DataFrame, formats: List[str], suffix: str = "", title: str = "Hint Recovery AUC", ylabel: str = "AUC", output_dir: Path = None) -> None:
    bias_categories = BIAS_ORDER
    x = range(len(bias_categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, model in enumerate(MODEL_ORDER):
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, None)
        offsets = [i + (idx - 1) * width for i in x]
        values = []
        for bias in bias_categories:
            val = per_model_bias[
                (per_model_bias["model"] == model)
                & (per_model_bias["bias_label"] == bias)
            ]["value"]
            values.append(valiloc(val))
        bars = ax.bar(offsets, values, width=width, label=label, color=color, edgecolor="black")

    ax.set_xticks([i for i in x])
    ax.set_xticklabels(bias_categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model")
    ax.set_title(title)
    fig.tight_layout()
    base_name = f"by_hint_type{suffix}"
    save_plot(fig, base_name, formats, output_dir)
    plt.close(fig)


def aggregate_by_model_dataset(best_auc_df: pd.DataFrame) -> pd.DataFrame:
    """Best metric per model/dataset averaged across bias types."""
    per_combo = (
        best_auc_df.groupby(["model", "dataset"])["value"].mean().reset_index()
    )
    per_combo["dataset_label"] = per_combo["dataset"].map({
        "aqua": "AQUA",
        "arc-challenge": "ARC-Challenge",
        "commonsense_qa": "CommonsenseQA",
        "mmlu": "MMLU",
    })
    per_combo["dataset_label"] = per_combo["dataset_label"].fillna(
        per_combo["dataset"].str.replace("_", " ").str.title()
    )
    return per_combo


def plot_model_dataset(per_combo: pd.DataFrame, formats: List[str], suffix: str = "", title: str = "Hint Recovery AUC", ylabel: str = "AUC", output_dir: Path = None) -> None:
    datasets = sorted(per_combo["dataset_label"].unique(), key=lambda d: ["AQUA","ARC-Challenge","CommonsenseQA","MMLU"].index(d) if d in ["AQUA","ARC-Challenge","CommonsenseQA","MMLU"] else d)
    x = range(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, model in enumerate(MODEL_ORDER):
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, None)
        offsets = [i + (idx - 1) * width for i in x]
        values = []
        for dataset in datasets:
            val = per_combo[
                (per_combo["model"] == model) & (per_combo["dataset_label"] == dataset)
            ]["value"]
            values.append(valiloc(val))
        bars = ax.bar(offsets, values, width=width, label=label, color=color, edgecolor="black")

    ax.set_xticks([i for i in x])
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model")
    ax.set_title(title)
    fig.tight_layout()
    base_name = f"by_dataset{suffix}"
    save_plot(fig, base_name, formats, output_dir)
    plt.close(fig)


def plot_combined(per_model_bias: pd.DataFrame, per_model_dataset: pd.DataFrame,
                  formats: List[str], title: str = "", ylabel: str = "AUC", output_dir: Path = None) -> None:
    """1x2 grid: (left) by dataset, (right) by hint type. Shared y-axis, single legend."""
    bias_categories = BIAS_ORDER
    datasets = sorted(
        per_model_dataset["dataset_label"].unique(),
        key=lambda d: ["AQUA", "ARC-Challenge", "CommonsenseQA", "MMLU"].index(d)
        if d in ["AQUA", "ARC-Challenge", "CommonsenseQA", "MMLU"] else d,
    )
    width = 0.25

    fig, (ax_ds, ax_bias) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left panel: by dataset
    x_ds = range(len(datasets))
    for idx, model in enumerate(MODEL_ORDER):
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, None)
        offsets = [i + (idx - 1) * width for i in x_ds]
        values = []
        for dataset in datasets:
            val = per_model_dataset[
                (per_model_dataset["model"] == model) & (per_model_dataset["dataset_label"] == dataset)
            ]["value"]
            values.append(valiloc(val))
        bars = ax_ds.bar(offsets, values, width=width, label=label, color=color, edgecolor="black")
    SHORT_LABELS = {"ARC-Challenge": "ARC", "CommonsenseQA": "CSQA"}
    ax_ds.set_xticks(list(x_ds))
    ax_ds.set_xticklabels([SHORT_LABELS.get(d, d) for d in datasets])
    ax_ds.set_ylabel(ylabel)
    ax_ds.set_ylim(0, 1.0)
    ax_ds.set_title(f"{title} across Datasets", fontsize=16)
    ax_ds.grid(axis="y", alpha=0.3)

    # Right panel: by hint type
    x_bias = range(len(bias_categories))
    for idx, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS.get(model, None)
        offsets = [i + (idx - 1) * width for i in x_bias]
        values = []
        for bias in bias_categories:
            val = per_model_bias[
                (per_model_bias["model"] == model) & (per_model_bias["bias_label"] == bias)
            ]["value"]
            values.append(valiloc(val))
        bars = ax_bias.bar(offsets, values, width=width, color=color, edgecolor="black")
    ax_bias.set_xticks(list(x_bias))
    ax_bias.set_xticklabels(bias_categories)
    ax_bias.set_title(f"{title} across Hint Types", fontsize=16)
    ax_bias.grid(axis="y", alpha=0.3)

    # Single legend inside the left panel
    handles, labels = ax_ds.get_legend_handles_labels()
    ax_ds.legend(handles, labels, loc="lower left", fontsize=11, title="Model")

    fig.tight_layout()

    save_plot(fig, "combined", formats, output_dir)
    plt.close(fig)


def valiloc(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.iloc[0])


TASK_LABELS = {
    "h_recovery": "Hint Recovery",
    "mot_vs_alg": "Motivated vs Aligned",
    "mot_vs_oth": "Preemptive Detection",
}

TASK_STEP_MODE = {
    "h_recovery": "last",
    "mot_vs_alg": "last",
    "mot_vs_oth": "first",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot probe AUC bar charts by hint type and dataset.")
    parser.add_argument(
        "--task",
        default="h_recovery",
        help="Probe task to plot (e.g., h_recovery, mot_vs_alg, mot_vs_oth). Default: h_recovery.",
    )
    parser.add_argument(
        "--step-mode",
        default=None,
        choices=["best", "first", "last"],
        help="Step selection mode. Default: task-dependent (last for h_recovery/mot_vs_alg, first for mot_vs_oth).",
    )
    parser.add_argument(
        "--best-layer",
        action="store_true",
        help="Use best layer per (model, dataset, bias) instead of last layer.",
    )
    parser.add_argument(
        "--fixed-best-layer",
        action="store_true",
        help="Find one best layer per model (averaged over datasets/biases) and use it for all.",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        choices=["auc", "accuracy"],
        help="Metric to plot (default: auc).",
    )
    parser.add_argument(
        "--fmt",
        action="append",
        default=None,
        help="Output formats (e.g., --fmt png --fmt pdf). Default: png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    formats = args.fmt if args.fmt else ["png"]
    task = args.task
    metric = args.metric
    step_mode = args.step_mode or TASK_STEP_MODE.get(task, "last")
    metric_label = "Accuracy" if metric == "accuracy" else "AUC"
    title = f"{TASK_LABELS.get(task, task)} {metric_label}"

    output_dir = FIGURES_DIR / task

    best_auc = load_best_auc_per_model(METRICS_DIR, task=task, step_mode=step_mode,
                                       use_best_layer=args.best_layer,
                                       fixed_best_layer=args.fixed_best_layer,
                                       metric=metric)

    per_model_bias = aggregate_by_bias(best_auc)
    plot_bias_detection(per_model_bias, formats, title=title, ylabel=metric_label, output_dir=output_dir)

    per_model_dataset = aggregate_by_model_dataset(best_auc)
    plot_model_dataset(per_model_dataset, formats, title=title, ylabel=metric_label, output_dir=output_dir)

    plot_combined(per_model_bias, per_model_dataset, formats, title=title, ylabel=metric_label, output_dir=output_dir)


if __name__ == "__main__":
    main()

