#!/usr/bin/env python3
"""
Compute bias-detection probe performance across bias types and models.

Reads the CSV logs in outputs/probe_metrics and produces a bar chart similar
to "Bias Detection — Bias Types", where each bar represents the best AUC
achieved by a model for detecting a given bias type averaged over datasets.
"""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Set larger font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
})


METRICS_DIR = Path("outputs/probe_metrics")
OUTPUT_DIR = Path("figures/bias_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}
BIAS_ORDER = ["Sycophancy", "Consistency", "Metadata"]

MODEL_LABELS = {
    "qwen-3-8b": "Qwen-3-8B-thinking",
    "llama-3.1-8b": "Llama-3.1-8B-Instruct",
    "gemma-3-4b": "Gemma-3-4B-it",
}
MODEL_ORDER = ["qwen-3-8b", "llama-3.1-8b", "gemma-3-4b"]
MODEL_COLORS = {
    "qwen-3-8b": "#8856a7",      # Qwen purple
    "llama-3.1-8b": "#e34a33",   # Llama red
    "gemma-3-4b": "#f4c20d",     # Gemma yellow
}


def load_best_auc_per_model(metrics_dir: Path, task: str = "bias", step_mode: str = "last") -> pd.DataFrame:
    """Load best AUC per model/dataset/bias from SQLite DB, falling back to CSVs.

    step_mode options:
      - "best": best AUC across all steps and layers (default)
      - "first": best AUC from the first step only
      - "last": best AUC from the last step only
    """
    # Try SQLite first
    db_path = metrics_dir.parent / "probe_metrics.db" if metrics_dir.name == "probe_metrics" else metrics_dir / "probe_metrics.db"
    if db_path.exists():
        return _load_best_auc_from_db(db_path, task, step_mode)
    return _load_best_auc_from_csvs(metrics_dir, task, step_mode)


def _load_best_auc_from_db(db_path: Path, task: str, step_mode: str) -> pd.DataFrame:
    """Load best AUC per model/dataset/bias from the SQLite database."""
    from thoughts.results_db import query_df
    df = query_df(
        "SELECT model, dataset, bias, layer, step, rfm_auc FROM probe_metrics WHERE probe = ?",
        params=(task,),
        db_path=str(db_path),
    )
    if df.empty:
        raise FileNotFoundError(f"No probe metrics for task={task} in {db_path}")
    records = []
    for (model, dataset, bias), group in df.groupby(["model", "dataset", "bias"]):
        if step_mode == "first":
            step_df = group[group["step"] == group["step"].min()]
        elif step_mode == "last":
            step_df = group[group["step"] == group["step"].max()]
        else:
            step_df = group
        best_row = step_df.loc[step_df["rfm_auc"].idxmax(), ["model", "dataset", "bias", "rfm_auc"]]
        records.append(best_row.to_frame().T)
    combined = pd.concat(records, ignore_index=True)
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
        best_row = step_df.loc[step_df["rfm_auc"].idxmax(), ["model", "dataset", "bias", "rfm_auc"]]
        records.append(best_row.to_frame().T)
    if not records:
        raise FileNotFoundError(f"No bias probe metrics found in {metrics_dir}")
    combined = pd.concat(records, ignore_index=True)
    combined["bias_label"] = combined["bias"].map(BIAS_LABELS)
    combined = combined.dropna(subset=["bias_label"])
    return combined


def aggregate_by_bias(best_auc_df: pd.DataFrame) -> pd.DataFrame:
    """Average best AUC over datasets for each model/bias."""
    per_model_bias = (
        best_auc_df.groupby(["model", "bias_label"])["rfm_auc"].mean().reset_index()
    )
    return per_model_bias


def save_plot(fig, base_name: str, formats: List[str]) -> None:
    for fmt in formats:
        out_path = OUTPUT_DIR / f"{base_name}.{fmt}"
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)


def plot_bias_detection(per_model_bias: pd.DataFrame, formats: List[str], suffix: str = "", title: str = "Hint Recovery AUC") -> None:
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
            ]["rfm_auc"]
            values.append(valiloc(val))
        bars = ax.bar(offsets, values, width=width, label=label, color=color, edgecolor="black")
        # Add value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks([i for i in x])
    ax.set_xticklabels(bias_categories)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Model")
    ax.set_title(title)
    fig.tight_layout()
    base_name = f"bias_detection_bias_types{suffix}"
    save_plot(fig, base_name, formats)
    plt.close(fig)


def aggregate_by_model_dataset(best_auc_df: pd.DataFrame) -> pd.DataFrame:
    """Best AUC per model/dataset averaged across bias types."""
    per_combo = (
        best_auc_df.groupby(["model", "dataset"])["rfm_auc"].mean().reset_index()
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


def plot_model_dataset(per_combo: pd.DataFrame, formats: List[str], suffix: str = "", title: str = "Hint Recovery AUC") -> None:
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
            ]["rfm_auc"]
            values.append(valiloc(val))
        bars = ax.bar(offsets, values, width=width, label=label, color=color, edgecolor="black")
        # Add value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks([i for i in x])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Model")
    ax.set_title(title)
    fig.tight_layout()
    base_name = f"bias_detection_model_dataset{suffix}"
    save_plot(fig, base_name, formats)
    plt.close(fig)


def valiloc(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.iloc[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Plot bias detection metrics.")
    parser.add_argument(
        "--fmt",
        action="append",
        default=["png"],
        help="Output formats (e.g., --fmt png --fmt pdf). Default png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    formats = args.fmt
    best_auc = load_best_auc_per_model(METRICS_DIR, task="bias")
    per_model_bias = aggregate_by_bias(best_auc)
    plot_bias_detection(per_model_bias, formats, suffix="", title="Hint Recovery AUC")

    per_model_dataset = aggregate_by_model_dataset(best_auc)
    plot_model_dataset(per_model_dataset, formats, suffix="", title="Hint Recovery AUC")

    # has-switched task
    switched_auc = load_best_auc_per_model(METRICS_DIR, task="has-switched")
    per_model_switched = aggregate_by_bias(switched_auc)
    plot_bias_detection(per_model_switched, formats, suffix="_has_switched", title="Post-Hoc Motivated Reasoning Detection")

    per_model_dataset_switched = aggregate_by_model_dataset(switched_auc)
    plot_model_dataset(per_model_dataset_switched, formats, suffix="_has_switched", title="Post-Hoc Motivated Reasoning Detection")

    # will-switch task (preemptive)
    will_auc = load_best_auc_per_model(METRICS_DIR, task="will-switch", step_mode="first")
    per_model_will = aggregate_by_bias(will_auc)
    plot_bias_detection(per_model_will, formats, suffix="_will_switch", title="Preemptive Motivated Reasoning Detection")

    per_model_dataset_will = aggregate_by_model_dataset(will_auc)
    plot_model_dataset(per_model_dataset_will, formats, suffix="_will_switch", title="Preemptive Motivated Reasoning Detection")


if __name__ == "__main__":
    main()

