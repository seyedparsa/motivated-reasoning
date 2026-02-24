#!/usr/bin/env python3
"""
Combined multi-task summary figure for motivated reasoning detection.

Creates a 1x3 subplot grid showing:
- Hint Recovery (bias task)
- Post-hoc Detection (has-switched task)
- Preemptive Detection (will-switch task)
"""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Set larger font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})

METRICS_DIR = Path("outputs/probe_metrics")
OUTPUT_DIR = Path("figures/bias_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "qwen-3-8b": "#8856a7",
    "llama-3.1-8b": "#e34a33",
    "gemma-3-4b": "#f4c20d",
}
MODEL_LABELS = {
    "qwen-3-8b": "Qwen-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gemma-3-4b": "Gemma-3-4B",
}
MODEL_ORDER = ["qwen-3-8b", "llama-3.1-8b", "gemma-3-4b"]

BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}
BIAS_ORDER = ["Sycophancy", "Consistency", "Metadata"]

TASK_CONFIG = {
    "bias": {
        "title": "Hint Recovery",
        "step_mode": "last",
    },
    "has-switched": {
        "title": "Post-hoc Detection",
        "step_mode": "last",
    },
    "will-switch": {
        "title": "Preemptive Detection",
        "step_mode": "first",
    },
}


def load_best_auc_from_csvs(metrics_dir: Path, task: str, step_mode: str) -> pd.DataFrame:
    """Load best AUC per model/dataset/bias from CSV files."""
    records: List[pd.DataFrame] = []
    for path in metrics_dir.glob(f"probe_metrics_*_{task}_per-step_3rel.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue

        # Filter by step_mode
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

        # Filter to RFM classifier if column exists, then get best row by AUC
        if "classifier" in step_df.columns:
            step_df = step_df[step_df["classifier"] == "rfm"]
        auc_col = "auc" if "auc" in step_df.columns else "rfm_auc"
        best_row = step_df.loc[step_df[auc_col].idxmax(), ["model", "dataset", "bias", auc_col]]
        if auc_col != "auc":
            best_row = best_row.rename({auc_col: "auc"})
        records.append(best_row.to_frame().T)

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    combined["bias_label"] = combined["bias"].map(BIAS_LABELS)
    combined = combined.dropna(subset=["bias_label"])
    return combined


def aggregate_by_bias(best_auc_df: pd.DataFrame) -> pd.DataFrame:
    """Average best AUC over datasets for each model/bias."""
    auc_col = "auc" if "auc" in best_auc_df.columns else "rfm_auc"
    per_model_bias = (
        best_auc_df.groupby(["model", "bias_label"])[auc_col].mean().reset_index()
    )
    if auc_col != "auc":
        per_model_bias = per_model_bias.rename(columns={auc_col: "auc"})
    return per_model_bias


def valiloc(series: pd.Series) -> float:
    """Safely get first value from series or return 0."""
    if series.empty:
        return 0.0
    return float(series.iloc[0])


def plot_combined_tasks(
    task_data: Dict[str, pd.DataFrame],
    output_path: Path,
    formats: List[str]
) -> None:
    """Create 1x3 subplot grid for all detection tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    task_order = ["bias", "has-switched", "will-switch"]

    for ax_idx, task in enumerate(task_order):
        ax = axes[ax_idx]
        config = TASK_CONFIG[task]
        per_model_bias = task_data.get(task)

        if per_model_bias is None or per_model_bias.empty:
            ax.set_title(f"{config['title']}\n(No data)")
            ax.set_ylim(0.5, 1.0)
            continue

        x = range(len(BIAS_ORDER))
        width = 0.25

        for idx, model in enumerate(MODEL_ORDER):
            label = MODEL_LABELS.get(model, model)
            color = MODEL_COLORS.get(model, "gray")
            offsets = [i + (idx - 1) * width for i in x]

            values = []
            for bias in BIAS_ORDER:
                val = per_model_bias[
                    (per_model_bias["model"] == model)
                    & (per_model_bias["bias_label"] == bias)
                ]["auc"]
                values.append(valiloc(val))

            bars = ax.bar(
                offsets, values, width=width,
                label=label if ax_idx == 0 else "",  # Only label first panel
                color=color, edgecolor="black"
            )

            # Add value labels on top
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.0%}",
                        ha="center", va="bottom", fontsize=8
                    )

        ax.set_xticks([i for i in x])
        ax.set_xticklabels(BIAS_ORDER, rotation=15, ha="right")
        ax.set_title(config["title"])
        ax.set_ylim(0.5, 1.05)

        # Add 0.5 baseline
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        if ax_idx == 0:
            ax.set_ylabel("AUC")

    # Add shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # Make room for legend

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot combined detection tasks.")
    parser.add_argument(
        "--fmt",
        action="append",
        default=None,
        help="Output formats (e.g., --fmt png --fmt pdf). Default: png.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=METRICS_DIR,
        help=f"Directory containing probe metrics CSVs. Default: {METRICS_DIR}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    formats = args.fmt if args.fmt else ["png"]
    metrics_dir = args.metrics_dir

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    # Load data for each task
    task_data = {}
    for task, config in TASK_CONFIG.items():
        best_auc = load_best_auc_from_csvs(metrics_dir, task, config["step_mode"])
        if not best_auc.empty:
            task_data[task] = aggregate_by_bias(best_auc)
            print(f"Loaded {len(best_auc)} records for {task}")
        else:
            print(f"Warning: No data found for task={task}")

    if not task_data:
        print("No data found for any task. Exiting.")
        return

    output_path = OUTPUT_DIR / "combined_detection_tasks"
    plot_combined_tasks(task_data, output_path, formats)


if __name__ == "__main__":
    main()
