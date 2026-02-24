#!/usr/bin/env python3
"""
Layer evolution line plots showing how probe performance varies across layers.

Creates line plots with x-axis = layer, y-axis = RFM AUC.
Three lines per plot: Beginning (dashed), Middle (dotted), End (solid).
"""
import argparse
import os
import sqlite3
from pathlib import Path
from typing import List

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

MOTIVATION_HOME = Path(os.environ.get("MOTIVATION_HOME", "/work/hdd/bbjr/pmirtaheri/motivated"))
OUTPUT_DIR = Path("figures/layer_evolution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "qwen-3-8b": "#8856a7",
    "llama-3.1-8b": "#e34a33",
    "gemma-3-4b": "#f4c20d",
}
MODEL_LABELS = {
    "qwen-3-8b": "Qwen-3-8B-thinking",
    "llama-3.1-8b": "Llama-3.1-8B-Instruct",
    "gemma-3-4b": "Gemma-3-4B-it",
}

STEP_STYLES = {
    0: {"linestyle": "--", "label": "Beginning", "alpha": 0.7},
    1: {"linestyle": ":", "label": "Middle", "alpha": 0.8},
    2: {"linestyle": "-", "label": "End", "alpha": 1.0},
}

BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}


def load_layer_data(db_path: Path, task: str = "has-switched") -> pd.DataFrame:
    """Load layer-wise probe performance from SQLite."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT model, dataset, bias, layer, step, auc
        FROM probe_metrics
        WHERE probe = ? AND classifier = 'rfm'
        ORDER BY model, dataset, bias, layer, step
    """
    df = pd.read_sql_query(query, conn, params=(task,))
    conn.close()
    return df


def plot_single_config(
    data: pd.DataFrame,
    model: str,
    dataset: str,
    bias: str,
    output_path: Path,
    formats: List[str]
) -> None:
    """Create layer evolution plot for a single (model, dataset, bias) config."""
    subset = data[
        (data["model"] == model)
        & (data["dataset"] == dataset)
        & (data["bias"] == bias)
    ]

    if subset.empty:
        print(f"No data for {model}/{dataset}/{bias}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for step in sorted(subset["step"].unique()):
        step_data = subset[subset["step"] == step].sort_values("layer")
        style = STEP_STYLES.get(step, {"linestyle": "-", "label": f"Step {step}", "alpha": 0.8})

        ax.plot(
            step_data["layer"],
            step_data["auc"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=style["label"],
            marker="o",
            markersize=6,
            linewidth=2
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("RFM AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(title="Position in CoT")
    ax.grid(True, alpha=0.3)

    model_label = MODEL_LABELS.get(model, model)
    bias_label = BIAS_LABELS.get(bias, bias)
    ax.set_title(f"{model_label} - {dataset} - {bias_label}")

    fig.tight_layout()

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)

    plt.close(fig)


def plot_aggregate(
    data: pd.DataFrame,
    output_path: Path,
    formats: List[str],
    task: str
) -> None:
    """Create aggregate layer evolution plot averaged across all configurations."""
    # Compute mean AUC across all (model, dataset, bias) for each (layer, step)
    agg = data.groupby(["layer", "step"])["auc"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    for step in sorted(agg["step"].unique()):
        step_data = agg[agg["step"] == step].sort_values("layer")
        style = STEP_STYLES.get(step, {"linestyle": "-", "label": f"Step {step}", "alpha": 0.8})

        ax.plot(
            step_data["layer"],
            step_data["auc"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=style["label"],
            marker="o",
            markersize=8,
            linewidth=2.5,
            color="#2c3e50"
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean RFM AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(title="Position in CoT")
    ax.grid(True, alpha=0.3)

    task_title = {
        "bias": "Hint Recovery",
        "has-switched": "Post-hoc Detection",
        "will-switch": "Preemptive Detection",
    }.get(task, task)
    ax.set_title(f"Layer Evolution - {task_title} (Aggregated)")

    fig.tight_layout()

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_by_model(
    data: pd.DataFrame,
    output_path: Path,
    formats: List[str],
    task: str
) -> None:
    """Create layer evolution plot with one line per model (final step only)."""
    # Filter to final step and aggregate by model/layer
    final_step = data["step"].max()
    final_data = data[data["step"] == final_step]
    agg = final_data.groupby(["model", "layer"])["auc"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODEL_COLORS.keys():
        model_data = agg[agg["model"] == model].sort_values("layer")
        if model_data.empty:
            continue

        ax.plot(
            model_data["layer"],
            model_data["auc"],
            linestyle="-",
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS.get(model, model)
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean RFM AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)

    task_title = {
        "bias": "Hint Recovery",
        "has-switched": "Post-hoc Detection",
        "will-switch": "Preemptive Detection",
    }.get(task, task)
    ax.set_title(f"Layer Evolution by Model - {task_title}")

    fig.tight_layout()

    for fmt in formats:
        out_file = output_path.parent / f"{output_path.stem}_by_model.{fmt}"
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_file, dpi=dpi)
        print(f"Saved: {out_file}")

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot layer evolution of probe performance.")
    parser.add_argument(
        "--fmt",
        action="append",
        default=None,
        help="Output formats (e.g., --fmt png --fmt pdf). Default: png.",
    )
    parser.add_argument(
        "--task",
        default="has-switched",
        choices=["has-switched", "bias", "will-switch"],
        help="Detection task to plot. Default: has-switched.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Create aggregate view averaged across all configurations.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to specific model (e.g., qwen-3-8b).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Filter to specific dataset (e.g., arc-challenge).",
    )
    parser.add_argument(
        "--bias",
        default=None,
        help="Filter to specific bias type (e.g., expert).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    formats = args.fmt if args.fmt else ["png"]

    db_path = MOTIVATION_HOME / "probe_metrics.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Probe metrics DB not found: {db_path}")

    data = load_layer_data(db_path, task=args.task)
    if data.empty:
        print(f"No data found for task={args.task}")
        return

    # Ensure layer and step are numeric
    data["layer"] = pd.to_numeric(data["layer"], errors="coerce")
    data["step"] = pd.to_numeric(data["step"], errors="coerce")
    data = data.dropna(subset=["layer", "step"])

    print(f"Loaded {len(data)} records for task={args.task}")
    print(f"Layers: {sorted(data['layer'].unique())}")
    print(f"Steps: {sorted(data['step'].unique())}")

    if args.aggregate:
        # Aggregate view
        output_path = OUTPUT_DIR / f"layer_evolution_{args.task}_aggregate"
        plot_aggregate(data, output_path, formats, args.task)
        plot_by_model(data, output_path, formats, args.task)
    elif args.model and args.dataset and args.bias:
        # Single configuration
        output_path = OUTPUT_DIR / f"{args.model}_{args.dataset}_{args.bias}"
        plot_single_config(data, args.model, args.dataset, args.bias, output_path, formats)
        print(f"Saved: {output_path}")
    else:
        # Generate all individual plots
        configs = data[["model", "dataset", "bias"]].drop_duplicates()
        for _, row in configs.iterrows():
            model, dataset, bias = row["model"], row["dataset"], row["bias"]
            output_path = OUTPUT_DIR / f"{model}_{dataset}_{bias}"
            plot_single_config(data, model, dataset, bias, output_path, formats)
        print(f"Generated {len(configs)} plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
