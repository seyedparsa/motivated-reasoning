#!/usr/bin/env python3
"""
Scatter plot comparing probe AUC vs LLM baseline AUC for motivated reasoning detection.

Points above the diagonal (y=x) indicate the probe outperforms CoT monitoring.
"""
import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set larger font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
})

MOTIVATION_HOME = Path(os.environ.get("MOTIVATION_HOME", "/work/hdd/bbjr/pmirtaheri/motivated"))
OUTPUT_DIR = Path("figures/bias_detection")
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
BIAS_MARKERS = {
    "expert": "o",      # circle
    "self": "s",        # square
    "metadata": "^",    # triangle
}
BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}
DATASET_ABBREV = {
    "aqua": "AQ",
    "arc-challenge": "ARC",
    "commonsense_qa": "CS",
    "mmlu": "MM",
}
DATASET_LABELS = {
    "aqua": "AQUA",
    "arc-challenge": "ARC-Challenge",
    "commonsense_qa": "CommonsenseQA",
    "mmlu": "MMLU",
}
DATASET_COLORS = {
    "aqua": "#1b9e77",
    "arc-challenge": "#d95f02",
    "commonsense_qa": "#7570b3",
    "mmlu": "#e7298a",
}


def load_probe_metrics(db_path: Path, task: str = "has-switched", mode: str = "best") -> pd.DataFrame:
    """Load probe AUC per (model, dataset, bias) from SQLite.

    mode options:
      - "best": best AUC across all layers and steps
      - "last_last": last layer, last step
      - "last_first": last layer, first step
      - "last_middle": last layer, middle step (step=1 for 3-checkpoint runs)
    """
    conn = sqlite3.connect(db_path)

    if mode == "best":
        query = """
            SELECT model, dataset, bias, MAX(rfm_auc) as probe_auc
            FROM probe_metrics
            WHERE probe = ?
            GROUP BY model, dataset, bias
        """
        df = pd.read_sql_query(query, conn, params=(task,))
    else:
        # Get all data first, then filter
        query = """
            SELECT model, dataset, bias, layer, step, rfm_auc
            FROM probe_metrics
            WHERE probe = ?
        """
        df = pd.read_sql_query(query, conn, params=(task,))

        if not df.empty:
            # Ensure layer and step are numeric
            df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
            df["step"] = pd.to_numeric(df["step"], errors="coerce")
            df = df.dropna(subset=["layer", "step"])

            records = []
            for (model, dataset, bias), group in df.groupby(["model", "dataset", "bias"]):
                max_layer = group["layer"].max()
                layer_df = group[group["layer"] == max_layer]

                if mode == "last_last":
                    target_step = layer_df["step"].max()
                elif mode == "last_first":
                    target_step = layer_df["step"].min()
                else:  # last_middle
                    steps = sorted(layer_df["step"].unique())
                    target_step = steps[len(steps) // 2]  # middle step

                row = layer_df[layer_df["step"] == target_step].iloc[0]
                records.append({
                    "model": model,
                    "dataset": dataset,
                    "bias": bias,
                    "probe_auc": row["rfm_auc"]
                })
            df = pd.DataFrame(records)

    conn.close()
    return df


def load_llm_metrics(db_path: Path, task: str = "has-switched") -> pd.DataFrame:
    """Load LLM baseline AUC from SQLite."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT model, dataset, bias, llm, llm_auc
        FROM llm_metrics
        WHERE probe = ?
    """
    df = pd.read_sql_query(query, conn, params=(task,))
    conn.close()
    return df


def merge_metrics(probe_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Join probe and LLM metrics on (model, dataset, bias)."""
    merged = pd.merge(
        probe_df, llm_df,
        on=["model", "dataset", "bias"],
        how="inner"
    )
    return merged


def plot_scatter(merged: pd.DataFrame, output_path: Path, formats: list) -> None:
    """Create scatter plot of probe vs LLM AUC."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot diagonal reference line
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, label="y = x", linewidth=1.5)

    # Plot each point with dataset label
    for _, row in merged.iterrows():
        model = row["model"]
        bias = row["bias"]
        dataset = row["dataset"]
        llm = row.get("llm", "")
        llm_auc = row["llm_auc"]
        probe_auc = row["probe_auc"]

        color = MODEL_COLORS.get(model, "gray")
        marker = BIAS_MARKERS.get(bias, "o")
        dataset_abbrev = DATASET_ABBREV.get(dataset, dataset[:2].upper())

        # Smaller size for gpt-5-nano, larger for gpt-5-mini
        size = 60 if llm == "gpt-5-nano" else 150

        ax.scatter(
            llm_auc, probe_auc,
            c=color, marker=marker,
            s=size, alpha=0.8, edgecolors="black", linewidth=0.5
        )

        # Add dataset label next to each point
        ax.annotate(
            dataset_abbrev,
            (llm_auc, probe_auc),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=7,
            alpha=0.7
        )

    # Create legend handles
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=10, label=MODEL_LABELS.get(m, m), markeredgecolor="black")
        for m, color in MODEL_COLORS.items()
    ]
    bias_handles = [
        plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray",
                   markersize=10, label=BIAS_LABELS.get(b, b), markeredgecolor="black")
        for b, marker in BIAS_MARKERS.items()
    ]

    # Create dataset legend text
    dataset_text = "  ".join([f"{abbr}={DATASET_LABELS.get(ds, ds)}"
                               for ds, abbr in DATASET_ABBREV.items()])

    # Create LLM size legend handles
    llm_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=6, label="gpt-5-nano", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=10, label="gpt-5-mini", markeredgecolor="black"),
    ]

    # Add legends
    legend1 = ax.legend(handles=model_handles, title="Model", loc="upper left")
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=bias_handles, title="Bias Type", loc="lower right")
    ax.add_artist(legend2)
    legend3 = ax.legend(handles=llm_handles, title="LLM Baseline", loc="center left",
                        bbox_to_anchor=(0, 0.35))
    ax.add_artist(legend3)

    # Add dataset key as text annotation
    ax.text(0.5, -0.12, f"Datasets: {dataset_text}",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    ax.set_xlabel("LLM Baseline AUC (CoT Monitoring)")
    ax.set_ylabel("Probe AUC (RFM)")
    ax.set_title("Post-Hoc Detection: Probe vs LLM Baseline")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add shaded region where probe > LLM
    ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                    alpha=0.1, color="green", label="Probe > LLM")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make room for dataset key

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_grid(merged_first: pd.DataFrame, merged_last: pd.DataFrame, output_path: Path, formats: list) -> None:
    """Create 2x3 grid: rows = first/last step, columns = models."""
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    row_data = [merged_first, merged_last]
    row_labels = ["First Step", "Last Step"]

    for row_idx, (data, row_label) in enumerate(zip(row_data, row_labels)):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_data = data[data["model"] == model]

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where probe > LLM
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points colored by dataset
            for dataset in model_data["dataset"].unique():
                dataset_data = model_data[model_data["dataset"] == dataset]
                color = DATASET_COLORS.get(dataset, "gray")
                label = DATASET_LABELS.get(dataset, dataset) if row_idx == 0 and col_idx == 0 else None
                ax.scatter(
                    dataset_data["llm_auc"], dataset_data["probe_auc"],
                    c=color, s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
                    label=label
                )

            ax.set_xlim(0.4, 1.0)
            ax.set_ylim(0.4, 1.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Column titles (model names) on top row only
            if row_idx == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=12)

            # Row labels on left column only
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nProbe AUC", fontsize=11)

            # X-axis label on bottom row only
            if row_idx == 1:
                ax.set_xlabel("LLM AUC", fontsize=11)

    # Add legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               title="Dataset", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Probe vs LLM Baseline (Last Layer)", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.12)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_simple(merged: pd.DataFrame, output_path: Path, formats: list) -> None:
    """Create simple scatter plot with one point per (model, dataset, bias), colored by model."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot diagonal reference line
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

    # Add shaded region where probe > LLM
    ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                    alpha=0.1, color="green")

    # Plot points colored by model
    for model in merged["model"].unique():
        model_data = merged[merged["model"] == model]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.scatter(
            model_data["llm_auc"], model_data["probe_auc"],
            c=color, s=100, alpha=0.8, edgecolors="black", linewidth=0.5,
            label=label
        )

    ax.set_xlabel("LLM Baseline AUC (CoT Monitoring)")
    ax.set_ylabel("Probe AUC (RFM)")
    ax.set_title("Probe vs LLM Baseline")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model", loc="upper left")

    fig.tight_layout()

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_single_model(merged: pd.DataFrame, model: str, output_path: Path, formats: list) -> None:
    """Create scatter plot of probe vs LLM AUC for a single model."""
    model_data = merged[merged["model"] == model]
    if model_data.empty:
        print(f"No data for model={model}")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot diagonal reference line
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, label="y = x", linewidth=1.5)

    # Add shaded region where probe > LLM
    ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                    alpha=0.1, color="green")

    # Plot each point colored by dataset
    for _, row in model_data.iterrows():
        bias = row["bias"]
        dataset = row["dataset"]
        llm = row.get("llm", "")
        llm_auc = row["llm_auc"]
        probe_auc = row["probe_auc"]

        color = DATASET_COLORS.get(dataset, "gray")
        marker = BIAS_MARKERS.get(bias, "o")

        # Hollow for gpt-5-nano, filled for gpt-5-mini
        if llm == "gpt-5-nano":
            ax.scatter(
                llm_auc, probe_auc,
                facecolors="none", edgecolors=color, marker=marker,
                s=120, alpha=0.9, linewidth=2
            )
        else:
            ax.scatter(
                llm_auc, probe_auc,
                c=color, marker=marker,
                s=120, alpha=0.8, edgecolors="black", linewidth=0.5
            )

    # Create legend handles
    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=10, label=DATASET_LABELS.get(ds, ds), markeredgecolor="black")
        for ds, color in DATASET_COLORS.items()
    ]
    bias_handles = [
        plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray",
                   markersize=10, label=BIAS_LABELS.get(b, b), markeredgecolor="black")
        for b, marker in BIAS_MARKERS.items()
    ]
    llm_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markersize=10, label="gpt-5-nano", markeredgecolor="gray", markeredgewidth=2),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=10, label="gpt-5-mini", markeredgecolor="black"),
    ]

    # Add legends
    legend1 = ax.legend(handles=dataset_handles, title="Dataset", loc="upper left")
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=bias_handles, title="Bias Type", loc="lower right")
    ax.add_artist(legend2)
    ax.legend(handles=llm_handles, title="LLM Baseline", loc="center left",
              bbox_to_anchor=(0, 0.38))

    ax.set_xlabel("LLM Baseline AUC (CoT Monitoring)")
    ax.set_ylabel("Probe AUC (RFM)")
    ax.set_title(f"Post-Hoc Detection: {MODEL_LABELS.get(model, model)}")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")

    plt.close(fig)


def print_summary(merged: pd.DataFrame) -> None:
    """Print summary statistics."""
    merged["delta"] = merged["probe_auc"] - merged["llm_auc"]
    print("\n=== Probe vs LLM Comparison ===")
    print(f"Total comparisons: {len(merged)}")
    print(f"Probe wins: {(merged['delta'] > 0).sum()}")
    print(f"LLM wins: {(merged['delta'] < 0).sum()}")
    print(f"Ties: {(merged['delta'] == 0).sum()}")
    print(f"Mean improvement: {merged['delta'].mean():.3f}")
    print(f"Max improvement: {merged['delta'].max():.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot probe vs LLM comparison.")
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
        "--by-model",
        action="store_true",
        help="Generate separate plots for each model.",
    )
    parser.add_argument(
        "--mode",
        default="best",
        choices=["best", "last_last", "last_first", "last_middle"],
        help="Layer/step selection: best (default), last_last, last_first, last_middle.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple plot with uniform markers (no color/shape variation).",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="Filter to specific LLM baseline (e.g., gpt-5-nano).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Generate 2x3 grid plot (rows=first/last step, cols=models).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    formats = args.fmt if args.fmt else ["png"]

    probe_db = MOTIVATION_HOME / "probe_metrics.db"
    llm_db = MOTIVATION_HOME / "llm_metrics.db"

    if not probe_db.exists():
        raise FileNotFoundError(f"Probe metrics DB not found: {probe_db}")
    if not llm_db.exists():
        raise FileNotFoundError(f"LLM metrics DB not found: {llm_db}")

    # Handle grid mode separately
    if args.grid:
        llm_df = load_llm_metrics(llm_db, task=args.task)

        # Load first step data
        probe_first = load_probe_metrics(probe_db, task=args.task, mode="last_first")
        merged_first = merge_metrics(probe_first, llm_df)
        if args.llm:
            merged_first = merged_first[merged_first["llm"] == args.llm]

        # Load last step data
        probe_last = load_probe_metrics(probe_db, task=args.task, mode="last_last")
        merged_last = merge_metrics(probe_last, llm_df)
        if args.llm:
            merged_last = merged_last[merged_last["llm"] == args.llm]

        llm_suffix = f"_{args.llm}" if args.llm else ""
        output_path = OUTPUT_DIR / f"probe_vs_llm_grid{llm_suffix}"
        plot_scatter_grid(merged_first, merged_last, output_path, formats)
        return

    probe_df = load_probe_metrics(probe_db, task=args.task, mode=args.mode)
    llm_df = load_llm_metrics(llm_db, task=args.task)
    merged = merge_metrics(probe_df, llm_df)

    # Filter by LLM if specified
    if args.llm:
        merged = merged[merged["llm"] == args.llm]

    if merged.empty:
        print(f"No matching data found for task={args.task}")
        return

    print(f"\nMode: {args.mode}")
    if args.llm:
        print(f"LLM filter: {args.llm}")
    print_summary(merged)

    # Build filename suffix based on mode
    mode_suffix = {
        "best": "_best",
        "last_last": "_last_layer_last_step",
        "last_first": "_last_layer_first_step",
        "last_middle": "_last_layer_middle_step",
    }[args.mode]

    # Add LLM suffix if filtered
    llm_suffix = f"_{args.llm}" if args.llm else ""

    if args.simple and args.by_model:
        # Simple plots, one per model
        for model in merged["model"].unique():
            model_data = merged[merged["model"] == model]
            model_safe = model.replace(".", "-")
            output_path = OUTPUT_DIR / f"probe_vs_llm_scatter_simple_{model_safe}{mode_suffix}{llm_suffix}"
            plot_scatter_simple(model_data, output_path, formats)
    elif args.simple:
        output_path = OUTPUT_DIR / f"probe_vs_llm_scatter_simple{mode_suffix}{llm_suffix}"
        plot_scatter_simple(merged, output_path, formats)
    elif args.by_model:
        # Generate separate plots for each model
        for model in merged["model"].unique():
            # Replace dots in model name to avoid path suffix issues
            model_safe = model.replace(".", "-")
            output_path = OUTPUT_DIR / f"probe_vs_llm_scatter_{model_safe}{mode_suffix}{llm_suffix}"
            plot_scatter_single_model(merged, model, output_path, formats)
    else:
        output_path = OUTPUT_DIR / f"probe_vs_llm_scatter{mode_suffix}{llm_suffix}"
        plot_scatter(merged, output_path, formats)


if __name__ == "__main__":
    main()
