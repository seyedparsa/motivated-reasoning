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


def load_probe_metrics(db_path: Path, task: str = "has-switched", mode: str = "best", filter_preemptive: bool = True,
                       balanced: int = 0, filter_mentions: int = 1) -> pd.DataFrame:
    """Load probe AUC per (model, dataset, bias) from SQLite.

    mode options:
      - "best": best AUC across all layers and steps
      - "last_last": last layer, last step
      - "last_first": last layer, first step
      - "last_middle": last layer, middle step (step=1 for 3-checkpoint runs)

    For preemptive tasks (mot_vs_*), only step=0 is used by default (set filter_preemptive=False to disable).
    """
    conn = sqlite3.connect(db_path)

    # For preemptive tasks, only use step=0 (before CoT generation) unless disabled
    is_preemptive = task.startswith("mot_vs_") and filter_preemptive
    step_filter = "AND step = 0" if is_preemptive else ""

    if mode == "best":
        query = f"""
            SELECT model, dataset, bias, MAX(auc) as probe_auc
            FROM probe_metrics
            WHERE probe = ? AND classifier = 'rfm' AND balanced = ? AND filter_mentions = ? {step_filter}
            GROUP BY model, dataset, bias
        """
        df = pd.read_sql_query(query, conn, params=(task, balanced, filter_mentions))
    else:
        # Get all data first, then filter
        query = f"""
            SELECT model, dataset, bias, layer, step, auc
            FROM probe_metrics
            WHERE probe = ? AND classifier = 'rfm' AND balanced = ? AND filter_mentions = ? {step_filter}
        """
        df = pd.read_sql_query(query, conn, params=(task, balanced, filter_mentions))

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
                    "probe_auc": row["auc"]
                })
            df = pd.DataFrame(records)

    conn.close()
    return df


def load_probe_metrics_by_step(db_path: Path, task: str, step: int, use_best_layer: bool = False, n_questions: int = None,
                               balanced: int = 0, filter_mentions: int = 1) -> pd.DataFrame:
    """Load probe AUC for a specific step.

    Args:
        use_best_layer: If True, use best layer (highest AUC). If False, use last layer.
        n_questions: If specified, filter to this n_questions value.
        balanced: Filter on balanced column (default 0).
        filter_mentions: Filter on filter_mentions column (default 1).
    """
    conn = sqlite3.connect(db_path)
    base_filter = "WHERE probe = ? AND classifier = 'rfm' AND step = ? AND balanced = ? AND filter_mentions = ?"
    base_params = [task, step, balanced, filter_mentions]
    if n_questions is not None:
        query = f"""
            SELECT model, dataset, bias, layer, auc
            FROM probe_metrics
            {base_filter} AND n_questions = ?
        """
        df = pd.read_sql_query(query, conn, params=base_params + [n_questions])
    else:
        query = f"""
            SELECT model, dataset, bias, layer, auc
            FROM probe_metrics
            {base_filter}
        """
        df = pd.read_sql_query(query, conn, params=base_params)
    conn.close()

    if df.empty:
        return df

    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df = df.dropna(subset=["layer"])

    records = []
    for (model, dataset, bias), group in df.groupby(["model", "dataset", "bias"]):
        if use_best_layer:
            row = group.loc[group["auc"].idxmax()]
        else:
            max_layer = group["layer"].max()
            row = group[group["layer"] == max_layer].iloc[0]
        records.append({
            "model": model,
            "dataset": dataset,
            "bias": bias,
            "probe_auc": row["auc"]
        })
    return pd.DataFrame(records)


def load_probe_metrics_by_scale(db_path: Path, task: str, step: int, use_best_layer: bool = False,
                                balanced: int = 0, filter_mentions: int = 1) -> pd.DataFrame:
    """Load probe AUC for a specific step, including n_questions for scale comparison."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT model, dataset, bias, layer, n_questions, auc
        FROM probe_metrics
        WHERE probe = ? AND classifier = 'rfm' AND step = ? AND balanced = ? AND filter_mentions = ?
    """
    df = pd.read_sql_query(query, conn, params=(task, step, balanced, filter_mentions))
    conn.close()

    if df.empty:
        return df

    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df = df.dropna(subset=["layer"])

    records = []
    for (model, dataset, bias, n_questions), group in df.groupby(["model", "dataset", "bias", "n_questions"]):
        if use_best_layer:
            row = group.loc[group["auc"].idxmax()]
        else:
            max_layer = group["layer"].max()
            row = group[group["layer"] == max_layer].iloc[0]
        records.append({
            "model": model,
            "dataset": dataset,
            "bias": bias,
            "n_questions": n_questions,
            "probe_auc": row["auc"]
        })
    return pd.DataFrame(records)


# Small vs large scale mappings
SCALE_MAPPING = {
    "arc-challenge": {"small": 800, "large": 900},
    "commonsense_qa": {"small": 3200, "large": 7500},
    "mmlu": {"small": 3200, "large": 8000},
}


def load_llm_metrics(db_path: Path, task: str = "has-switched", balanced: int = 0, filter_mentions: int = 1) -> pd.DataFrame:
    """Load LLM baseline AUC from SQLite."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT model, dataset, bias, llm, llm_auc
        FROM llm_metrics
        WHERE probe = ? AND balanced = ? AND filter_mentions = ?
    """
    df = pd.read_sql_query(query, conn, params=(task, balanced, filter_mentions))
    conn.close()
    return df


import warnings


def check_duplicates(merged_by_step: dict, group_cols: list = None) -> None:
    """Warn if any (model, dataset, bias) group has more than one datapoint per step."""
    if group_cols is None:
        group_cols = ["model", "dataset", "bias"]
    for step, df in merged_by_step.items():
        if df.empty:
            continue
        counts = df.groupby(group_cols).size()
        dupes = counts[counts > 1]
        if len(dupes) > 0:
            warnings.warn(
                f"Step {step}: found duplicate datapoints for {len(dupes)} group(s):\n{dupes.to_string()}"
            )


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
    # Title varies by task type
    task = output_path.stem.split("_")[-1] if "mot_vs" in str(output_path) else "post-hoc"
    title_prefix = "Preemptive" if "mot_vs" in str(output_path) else "Post-Hoc"
    ax.set_title(f"{title_prefix} Detection: Probe vs LLM Baseline")
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


def plot_scatter_grid_by_step(merged_by_step: dict, output_path: Path, formats: list, title: str = "Probe vs LLM Baseline") -> None:
    """Create 2x3 grid: rows = steps (beginning, last), columns = models."""
    check_duplicates(merged_by_step)
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    step_labels = {0: "Preemptive", 2: "Post-Hoc"}
    steps = [0, 2]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)

    for row_idx, step in enumerate(steps):
        merged = merged_by_step.get(step, pd.DataFrame())
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_data = merged[merged["model"] == model] if not merged.empty else pd.DataFrame()

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where probe > LLM
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points colored by dataset
            if not model_data.empty:
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

            # Row labels (step names) on left column only
            if col_idx == 0:
                ax.set_ylabel(f"{step_labels[step]}\nProbe AUC", fontsize=11)

            # X-axis label on bottom row only
            if row_idx == 1:
                ax.set_xlabel("LLM AUC", fontsize=11)

    # Add legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               title="Dataset", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.12)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_grid_all_biases(all_bias_data: dict, output_path: Path, formats: list, title: str = "Probe vs LLM Baseline", steps: list = None) -> None:
    """Create 3x3 grid: rows = hint types, cols = models. Preemptive and post-hoc steps overlap in same subplot.

    Points colored by dataset. Preemptive = circle (hollow), Post-Hoc = circle (filled).
    """
    if steps is None:
        steps = [0, 2]
    plot_both = (0 in steps and 2 in steps)
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    bias_order = ["expert", "self", "metadata"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)

    for row_idx, bias in enumerate(bias_order):
        merged_by_step = all_bias_data.get(bias, {})
        merged_begin = merged_by_step.get(0, pd.DataFrame()) if 0 in steps else pd.DataFrame()
        merged_end = merged_by_step.get(2, pd.DataFrame()) if 2 in steps else pd.DataFrame()
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            begin_data = merged_begin[merged_begin["model"] == model] if not merged_begin.empty else pd.DataFrame()
            end_data = merged_end[merged_end["model"] == model] if not merged_end.empty else pd.DataFrame()

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where probe > LLM
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points for each dataset
            for dataset in DATASET_COLORS.keys():
                color = DATASET_COLORS[dataset]
                b_rows = begin_data[begin_data["dataset"] == dataset] if not begin_data.empty else pd.DataFrame()
                e_rows = end_data[end_data["dataset"] == dataset] if not end_data.empty else pd.DataFrame()

                # Dataset label only in first subplot
                ds_label = DATASET_LABELS.get(dataset, dataset) if row_idx == 0 and col_idx == 0 else None

                if plot_both:
                    # Preemptive: hollow circle
                    if not b_rows.empty:
                        ax.scatter(
                            b_rows["llm_auc"], b_rows["probe_auc"],
                            facecolors="none", edgecolors=color, marker="o",
                            s=80, alpha=0.9, linewidth=2,
                            label=ds_label
                        )
                    # Post-Hoc: filled circle
                    if not e_rows.empty:
                        ax.scatter(
                            e_rows["llm_auc"], e_rows["probe_auc"],
                            c=color, marker="o",
                            s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
                        )
                else:
                    # Single step: filled circle
                    data = b_rows if not b_rows.empty else e_rows
                    if not data.empty:
                        ax.scatter(
                            data["llm_auc"], data["probe_auc"],
                            c=color, marker="o",
                            s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
                            label=ds_label
                        )

            ax.set_xlim(0.4, 1.0)
            ax.set_ylim(0.4, 1.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Column titles (model names) on top row
            if row_idx == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=12)

            # Row labels (hint type) on left column
            if col_idx == 0:
                bias_label = BIAS_LABELS.get(bias, bias)
                ax.set_ylabel(f"{bias_label}\nProbe AUC", fontsize=11)

            # X-axis label on bottom row
            if row_idx == 2:
                ax.set_xlabel("LLM AUC", fontsize=11)

    # Build legend with all datasets always shown
    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=9, label=DATASET_LABELS.get(ds, ds), markeredgecolor="black")
        for ds, color in DATASET_COLORS.items()
    ]
    handles = dataset_handles
    labels = [DATASET_LABELS.get(ds, ds) for ds in DATASET_COLORS.keys()]
    if plot_both:
        step_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markersize=9, label="Preemptive", markeredgecolor="gray", markeredgewidth=2),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=9, label="Post-Hoc", markeredgecolor="black"),
        ]
        handles += step_handles
        labels += ["Preemptive", "Post-Hoc"]
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.08)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_grid_all_datasets(all_dataset_data: dict, output_path: Path, formats: list, title: str = "Probe vs LLM Baseline", steps: list = None) -> None:
    """Create 4x3 grid: rows = datasets, cols = models. Preemptive and post-hoc steps overlap in same subplot.

    Points colored by hint type. Preemptive = circle (hollow), Post-Hoc = circle (filled).
    """
    if steps is None:
        steps = [0, 2]
    plot_both = (0 in steps and 2 in steps)
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    dataset_order = ["aqua", "arc-challenge", "commonsense_qa", "mmlu"]

    BIAS_COLORS = {
        "expert": "#1b9e77",
        "self": "#d95f02",
        "metadata": "#7570b3",
    }

    fig, axes = plt.subplots(4, 3, figsize=(10, 13), sharex=True, sharey=True)

    for row_idx, dataset in enumerate(dataset_order):
        merged_by_step = all_dataset_data.get(dataset, {})
        merged_begin = merged_by_step.get(0, pd.DataFrame()) if 0 in steps else pd.DataFrame()
        merged_end = merged_by_step.get(2, pd.DataFrame()) if 2 in steps else pd.DataFrame()
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            begin_data = merged_begin[merged_begin["model"] == model] if not merged_begin.empty else pd.DataFrame()
            end_data = merged_end[merged_end["model"] == model] if not merged_end.empty else pd.DataFrame()

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where probe > LLM
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points for each bias type
            for bias in BIAS_COLORS.keys():
                color = BIAS_COLORS[bias]
                b_rows = begin_data[begin_data["bias"] == bias] if not begin_data.empty else pd.DataFrame()
                e_rows = end_data[end_data["bias"] == bias] if not end_data.empty else pd.DataFrame()

                # Bias label only in first subplot
                bias_label = BIAS_LABELS.get(bias, bias) if row_idx == 0 and col_idx == 0 else None

                if plot_both:
                    # Preemptive: hollow circle
                    if not b_rows.empty:
                        ax.scatter(
                            b_rows["llm_auc"], b_rows["probe_auc"],
                            facecolors="none", edgecolors=color, marker="o",
                            s=80, alpha=0.9, linewidth=2,
                            label=bias_label
                        )
                    # Post-Hoc: filled circle
                    if not e_rows.empty:
                        ax.scatter(
                            e_rows["llm_auc"], e_rows["probe_auc"],
                            c=color, marker="o",
                            s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
                        )
                else:
                    # Single step: filled circle
                    data = b_rows if not b_rows.empty else e_rows
                    if not data.empty:
                        ax.scatter(
                            data["llm_auc"], data["probe_auc"],
                            c=color, marker="o",
                            s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
                            label=bias_label
                        )

            ax.set_xlim(0.4, 1.0)
            ax.set_ylim(0.4, 1.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Column titles (model names) on top row
            if row_idx == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=12)

            # Row labels (dataset) on left column
            if col_idx == 0:
                ds_label = DATASET_LABELS.get(dataset, dataset)
                ax.set_ylabel(f"{ds_label}\nProbe AUC", fontsize=11)

            # X-axis label on bottom row
            if row_idx == 3:
                ax.set_xlabel("LLM AUC", fontsize=11)

    # Build legend with all bias types always shown
    bias_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=9, label=BIAS_LABELS.get(b, b), markeredgecolor="black")
        for b, color in BIAS_COLORS.items()
    ]
    handles = bias_handles
    labels = [BIAS_LABELS.get(b, b) for b in BIAS_COLORS.keys()]
    if plot_both:
        step_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markersize=9, label="Preemptive", markeredgecolor="gray", markeredgewidth=2),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=9, label="Post-Hoc", markeredgecolor="black"),
        ]
        handles += step_handles
        labels += ["Preemptive", "Post-Hoc"]
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.06)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_bar_grid(probe_by_step: dict, output_path: Path, formats: list, title: str = "Probe AUC", steps: list = None) -> None:
    """Create 4x3 bar chart grid: rows = datasets, cols = models. Bars = hint types, grouped by step."""
    import numpy as np

    if steps is None:
        steps = [0, 2]
    plot_both = (0 in steps and 2 in steps)
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    dataset_order = ["aqua", "arc-challenge", "commonsense_qa", "mmlu"]
    bias_order = ["expert", "self", "metadata"]

    BIAS_COLORS = {
        "expert": "#1b9e77",
        "self": "#d95f02",
        "metadata": "#7570b3",
    }
    step_labels = {0: "Preemptive", 2: "Post-Hoc"}

    fig, axes = plt.subplots(4, 3, figsize=(12, 13), sharey=True)

    for row_idx, dataset in enumerate(dataset_order):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]

            if plot_both:
                n_groups = len(bias_order)
                n_bars = len(steps)
                bar_width = 0.35
                x = np.arange(n_groups)

                for step_idx, step in enumerate(steps):
                    df = probe_by_step.get(step, pd.DataFrame())
                    subset = df[(df["model"] == model) & (df["dataset"] == dataset)] if not df.empty else pd.DataFrame()

                    values = []
                    for bias in bias_order:
                        row = subset[subset["bias"] == bias] if not subset.empty else pd.DataFrame()
                        values.append(row["probe_auc"].values[0] if not row.empty else 0)

                    offset = (step_idx - 0.5) * bar_width + bar_width / 2
                    colors = [BIAS_COLORS[b] for b in bias_order]
                    alpha = 0.5 if step == 0 else 0.9
                    bars = ax.bar(x + offset, values, bar_width, color=colors, alpha=alpha,
                                  edgecolor="black", linewidth=0.5)
            else:
                step = steps[0]
                df = probe_by_step.get(step, pd.DataFrame())
                subset = df[(df["model"] == model) & (df["dataset"] == dataset)] if not df.empty else pd.DataFrame()

                values = []
                for bias in bias_order:
                    row = subset[subset["bias"] == bias] if not subset.empty else pd.DataFrame()
                    values.append(row["probe_auc"].values[0] if not row.empty else 0)

                x = np.arange(len(bias_order))
                colors = [BIAS_COLORS[b] for b in bias_order]
                ax.bar(x, values, 0.6, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)

            ax.set_ylim(0.4, 1.0)
            ax.set_xticks(np.arange(len(bias_order)))
            ax.set_xticklabels([BIAS_LABELS.get(b, b) for b in bias_order], fontsize=9, rotation=30, ha="right")
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.grid(True, axis="y", alpha=0.3)

            # Column titles on top row
            if row_idx == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=12)

            # Row labels on left column
            if col_idx == 0:
                ds_label = DATASET_LABELS.get(dataset, dataset)
                ax.set_ylabel(f"{ds_label}\nProbe AUC", fontsize=11)

    # Build legend
    bias_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=BIAS_COLORS[b],
                   markersize=10, label=BIAS_LABELS.get(b, b), markeredgecolor="black")
        for b in bias_order
    ]
    handles = bias_handles
    labels = [BIAS_LABELS.get(b, b) for b in bias_order]
    if plot_both:
        from matplotlib.patches import Patch
        step_handles = [
            Patch(facecolor="gray", alpha=0.5, edgecolor="black", label="Preemptive"),
            Patch(facecolor="gray", alpha=0.9, edgecolor="black", label="Post-Hoc"),
        ]
        handles += step_handles
        labels += ["Preemptive", "Post-Hoc"]
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.06)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_avg_over_datasets(merged_by_step: dict, output_path: Path, formats: list, title: str = "Probe vs LLM Baseline", steps: list = None) -> None:
    """Create 1x3 grid: cols = models. Points colored by hint type, averaged over datasets.

    Preemptive = circle (hollow), Post-Hoc = circle (filled).
    """
    if steps is None:
        steps = [0, 2]
    plot_both = (0 in steps and 2 in steps)
    models = ["qwen-3-8b", "llama-3.1-8b", "gemma-3-4b"]

    BIAS_COLORS = {
        "expert": "#1b9e77",
        "self": "#d95f02",
        "metadata": "#7570b3",
    }

    axis_min = 0.5
    axis_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)

    for col_idx, model in enumerate(models):
        ax = axes[col_idx]

        # Plot diagonal reference line
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--", alpha=0.4, linewidth=1.5)

        # Shaded region: green above diagonal (probe better)
        ax.fill_between([axis_min, axis_max], [axis_min, axis_max], [axis_max, axis_max],
                        alpha=0.08, color="green")

        # Directional arrows
        arrow_kw = dict(arrowstyle="->", color="gray", lw=1.5)
        ax.annotate("", xy=(0.72, 0.57), xytext=(0.60, 0.57), arrowprops=arrow_kw)
        ax.annotate("", xy=(0.57, 0.72), xytext=(0.57, 0.60), arrowprops=arrow_kw)
        if 0 in steps:
            ax.text(0.60, 0.555, r"LLM Better $\mathbf{(Needs\ CoT)}$", ha="left", va="top", fontsize=9, color="gray")
            ax.text(0.555, 0.60, r"Probe Better $\mathbf{(No\ CoT\ Generation)}$", ha="right", va="bottom", fontsize=9, color="gray", rotation=90)
        else:
            ax.text(0.60, 0.555, "LLM Better", ha="left", va="top", fontsize=9, color="gray")
            ax.text(0.555, 0.60, "Probe Better", ha="right", va="bottom", fontsize=9, color="gray", rotation=90)

        for bias in BIAS_COLORS.keys():
            color = BIAS_COLORS[bias]
            bias_label = BIAS_LABELS.get(bias, bias) if col_idx == 0 else None

            for step in steps:
                merged = merged_by_step.get(step, pd.DataFrame())
                if merged.empty:
                    continue
                subset = merged[(merged["model"] == model) & (merged["bias"] == bias)]
                if subset.empty:
                    continue

                avg_llm = subset["llm_auc"].mean()
                avg_probe = subset["probe_auc"].mean()
                if plot_both:
                    if step == 0:
                        ax.scatter(
                            avg_llm, avg_probe,
                            facecolors="none", edgecolors=color, marker="o",
                            s=150, alpha=0.9, linewidth=2.5,
                            label=bias_label, zorder=5
                        )
                    else:
                        ax.scatter(
                            avg_llm, avg_probe,
                            c=color, marker="o",
                            s=150, alpha=0.9, edgecolors="white", linewidth=1.5,
                            zorder=5
                        )
                else:
                    ax.scatter(
                        avg_llm, avg_probe,
                        c=color, marker="o",
                        s=150, alpha=0.9, edgecolors="white", linewidth=1.5,
                        label=bias_label, zorder=5
                    )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=13)
        ax.set_xlabel("LLM AUC", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Probe AUC", fontsize=12)

    # Build legend with all bias types always shown
    BIAS_COLORS_AVG = {
        "expert": "#1b9e77",
        "self": "#d95f02",
        "metadata": "#7570b3",
    }
    bias_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=10, label=BIAS_LABELS.get(b, b), markeredgecolor="black")
        for b, color in BIAS_COLORS_AVG.items()
    ]
    handles = bias_handles
    labels = [BIAS_LABELS.get(b, b) for b in BIAS_COLORS_AVG.keys()]
    if plot_both:
        step_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markersize=10, label="Preemptive", markeredgecolor="gray", markeredgewidth=2),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=10, label="Post-Hoc", markeredgecolor="black"),
        ]
        handles += step_handles
        labels += ["Preemptive", "Post-Hoc"]
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.22)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_avg_over_biases(merged_by_step: dict, output_path: Path, formats: list, title: str = "Probe vs LLM Baseline", steps: list = None) -> None:
    """Create 1x3 grid: cols = models. Points colored by dataset, averaged over hint types.

    Preemptive = circle (hollow), Post-Hoc = circle (filled).
    """
    if steps is None:
        steps = [0, 2]
    plot_both = (0 in steps and 2 in steps)
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)

    for col_idx, model in enumerate(models):
        ax = axes[col_idx]

        # Plot diagonal reference line
        ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

        # Add shaded region where probe > LLM
        ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                        alpha=0.1, color="green")

        for dataset in DATASET_COLORS.keys():
            color = DATASET_COLORS[dataset]
            ds_label = DATASET_LABELS.get(dataset, dataset) if col_idx == 0 else None

            for step in steps:
                merged = merged_by_step.get(step, pd.DataFrame())
                if merged.empty:
                    continue
                subset = merged[(merged["model"] == model) & (merged["dataset"] == dataset)]
                if subset.empty:
                    continue

                avg_llm = subset["llm_auc"].mean()
                avg_probe = subset["probe_auc"].mean()

                if plot_both:
                    if step == 0:
                        ax.scatter(
                            avg_llm, avg_probe,
                            facecolors="none", edgecolors=color, marker="o",
                            s=100, alpha=0.9, linewidth=2,
                            label=ds_label
                        )
                    else:
                        ax.scatter(
                            avg_llm, avg_probe,
                            c=color, marker="o",
                            s=100, alpha=0.8, edgecolors="black", linewidth=0.5,
                        )
                else:
                    ax.scatter(
                        avg_llm, avg_probe,
                        c=color, marker="o",
                        s=100, alpha=0.8, edgecolors="black", linewidth=0.5,
                        label=ds_label
                    )

        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12)
        if len(steps) == 1 and steps[0] == 0:
            xlabel = "LLM AUC"
            ylabel = "Probe AUC"
        elif len(steps) == 1 and steps[0] == 2:
            xlabel = "LLM AUC"
            ylabel = "Probe AUC (with access to CoT)"
        else:
            xlabel = "LLM AUC"
            ylabel = "Probe AUC"
        ax.set_xlabel(xlabel, fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)

        # Directional arrows
        arrow_kw = dict(arrowstyle="->", color="gray", lw=1.5)
        ax.annotate("", xy=(0.66, 0.48), xytext=(0.52, 0.48), arrowprops=arrow_kw)
        ax.text(0.50, 0.47, "LLM Better\n(Needs CoT)", ha="left", va="top", fontsize=8, color="gray")
        ax.annotate("", xy=(0.48, 0.66), xytext=(0.48, 0.52), arrowprops=arrow_kw)
        ax.text(0.47, 0.50, "Probe Better\n(No Generation)", ha="right", va="bottom", fontsize=8, color="gray", rotation=90)

    # Build legend with all datasets always shown
    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=9, label=DATASET_LABELS.get(ds, ds), markeredgecolor="black")
        for ds, color in DATASET_COLORS.items()
    ]
    handles = dataset_handles
    labels = [DATASET_LABELS.get(ds, ds) for ds in DATASET_COLORS.keys()]
    if plot_both:
        step_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markersize=9, label="Preemptive", markeredgecolor="gray", markeredgewidth=2),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=9, label="Post-Hoc", markeredgecolor="black"),
        ]
        handles += step_handles
        labels += ["Preemptive", "Post-Hoc"]
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=10,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.15)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scatter_grid_by_step_color_bias(merged_by_step: dict, output_path: Path, formats: list, title: str = "Probe vs LLM Baseline") -> None:
    """Create 2x3 grid: rows = steps (beginning, last), columns = models. Points colored by bias type."""
    check_duplicates(merged_by_step)
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    step_labels = {0: "Preemptive", 2: "Post-Hoc"}
    steps = [0, 2]

    BIAS_COLORS = {
        "expert": "#1b9e77",
        "self": "#d95f02",
        "metadata": "#7570b3",
    }

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)

    for row_idx, step in enumerate(steps):
        merged = merged_by_step.get(step, pd.DataFrame())
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_data = merged[merged["model"] == model] if not merged.empty else pd.DataFrame()

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where probe > LLM
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points colored by bias type
            if not model_data.empty:
                for bias in model_data["bias"].unique():
                    bias_data = model_data[model_data["bias"] == bias]
                    color = BIAS_COLORS.get(bias, "gray")
                    label = BIAS_LABELS.get(bias, bias) if row_idx == 0 and col_idx == 0 else None
                    ax.scatter(
                        bias_data["llm_auc"], bias_data["probe_auc"],
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

            # Row labels (step names) on left column only
            if col_idx == 0:
                ax.set_ylabel(f"{step_labels[step]}\nProbe AUC", fontsize=11)

            # X-axis label on bottom row only
            if row_idx == 1:
                ax.set_xlabel("LLM AUC", fontsize=11)

    # Add legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               title="Hint Type", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.12)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_probe_vs_probe_grid(merged_by_step: dict, output_path: Path, formats: list,
                              xlabel: str = "Post-Hoc Probe AUC",
                              ylabel: str = "Preemptive Probe AUC",
                              title: str = "Motivated vs Aligned vs Motivated vs Others") -> None:
    """Create 2x3 grid comparing two probes: rows = steps (beginning, last), columns = models."""
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    step_labels = {0: "Preemptive", 2: "Post-Hoc"}
    steps = [0, 2]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)

    for row_idx, step in enumerate(steps):
        merged = merged_by_step.get(step, pd.DataFrame())
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_data = merged[merged["model"] == model] if not merged.empty else pd.DataFrame()

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where y > x
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points colored by dataset
            if not model_data.empty:
                for dataset in model_data["dataset"].unique():
                    dataset_data = model_data[model_data["dataset"] == dataset]
                    color = DATASET_COLORS.get(dataset, "gray")
                    label = DATASET_LABELS.get(dataset, dataset) if row_idx == 0 and col_idx == 0 else None
                    ax.scatter(
                        dataset_data["probe_auc_x"], dataset_data["probe_auc_y"],
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

            # Row labels (step names) on left column only
            if col_idx == 0:
                ax.set_ylabel(f"{step_labels[step]}\n{ylabel}", fontsize=11)

            # X-axis label on bottom row only
            if row_idx == 1:
                ax.set_xlabel(xlabel, fontsize=11)

    # Add legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               title="Dataset", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.12)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_scale_comparison_grid(merged_by_step: dict, output_path: Path, formats: list,
                                title: str = "Small Scale vs Large Scale") -> None:
    """Create 2x3 grid comparing small vs large scale: rows = steps (beginning, last), columns = models."""
    models = ["gemma-3-4b", "llama-3.1-8b", "qwen-3-8b"]
    step_labels = {0: "Preemptive", 2: "Post-Hoc"}
    steps = [0, 2]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)

    for row_idx, step in enumerate(steps):
        merged = merged_by_step.get(step, pd.DataFrame())
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_data = merged[merged["model"] == model] if not merged.empty else pd.DataFrame()

            # Plot diagonal reference line
            ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.5, linewidth=1.5)

            # Add shaded region where large > small
            ax.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                            alpha=0.1, color="green")

            # Plot points colored by dataset
            if not model_data.empty:
                for dataset in model_data["dataset"].unique():
                    dataset_data = model_data[model_data["dataset"] == dataset]
                    color = DATASET_COLORS.get(dataset, "gray")
                    label = DATASET_LABELS.get(dataset, dataset) if row_idx == 0 and col_idx == 0 else None
                    ax.scatter(
                        dataset_data["small_auc"], dataset_data["large_auc"],
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

            # Row labels (step names) on left column only
            if col_idx == 0:
                ax.set_ylabel(f"{step_labels[step]}\nLarge Scale AUC", fontsize=11)

            # X-axis label on bottom row only
            if row_idx == 1:
                ax.set_xlabel("Small Scale AUC", fontsize=11)

    # Add legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               title="Dataset", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.12)

    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt.lower() == "png" else 300
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
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
    title_prefix = "Preemptive" if "mot_vs" in str(output_path) else "Post-Hoc"
    ax.set_title(f"{title_prefix} Detection: {MODEL_LABELS.get(model, model)}")
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
        help="Detection task/probe to plot (e.g., has-switched, mot_vs_oth). Default: has-switched.",
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
        default="gpt-5-nano",
        help="Filter to specific LLM baseline (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--llm-task",
        default=None,
        help="Use LLM metrics from a different task (e.g., has-switched for preemptive comparison).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Generate 2x3 grid plot (rows=first/last step, cols=models).",
    )
    parser.add_argument(
        "--compare-probes",
        nargs=2,
        metavar=("PROBE_X", "PROBE_Y"),
        help="Compare two probes directly (no LLM). E.g., --compare-probes has-switched mot_vs_oth",
    )
    parser.add_argument(
        "--best-layer",
        action="store_true",
        help="Use best layer (highest AUC) instead of last layer.",
    )
    parser.add_argument(
        "--compare-scales",
        action="store_true",
        help="Compare small scale vs large scale for the same (model, dataset, bias).",
    )
    parser.add_argument(
        "--bias",
        default=None,
        help="Filter to specific bias type (e.g., expert, self, metadata). Use 'each' to generate one plot per bias type.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Filter to specific dataset(s). Comma-separated for multiple (e.g., commonsense_qa,mmlu). Use 'each' for one plot per dataset.",
    )
    parser.add_argument(
        "--combined-biases",
        action="store_true",
        help="Generate a single 3x3 grid with all hint types as rows, preemptive+post-hoc overlaid.",
    )
    parser.add_argument(
        "--combined-datasets",
        action="store_true",
        help="Generate a single 4x3 grid with all datasets as rows, colored by hint type, preemptive+post-hoc overlaid.",
    )
    parser.add_argument(
        "--avg-datasets",
        action="store_true",
        help="Average over datasets (equally weighted). Generates 1x3 grid colored by hint type.",
    )
    parser.add_argument(
        "--avg-biases",
        action="store_true",
        help="Average over hint types (equally weighted). Generates 1x3 grid colored by dataset.",
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Generate bar chart grid (4x3: rows=datasets, cols=models, bars=hint types).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Only plot a specific step (0=preemptive, 2=post-hoc). Default: plot both.",
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

    # Handle probe vs probe comparison
    if args.compare_probes:
        probe_x, probe_y = args.compare_probes
        use_best = getattr(args, 'best_layer', False)
        merged_by_step = {}
        for step in [0, 1, 2]:
            df_x = load_probe_metrics_by_step(probe_db, task=probe_x, step=step, use_best_layer=use_best)
            df_y = load_probe_metrics_by_step(probe_db, task=probe_y, step=step, use_best_layer=use_best)
            if not df_x.empty and not df_y.empty:
                merged = pd.merge(
                    df_x, df_y,
                    on=["model", "dataset", "bias"],
                    how="inner",
                    suffixes=("_x", "_y")
                )
                merged_by_step[step] = merged
            else:
                merged_by_step[step] = pd.DataFrame()

        # Determine labels
        probe_labels = {
            "has-switched": "Motivated vs Aligned",
            "mot_vs_oth": "Motivated vs Others",
            "will-switch": "Will Switch",
        }
        xlabel = probe_labels.get(probe_x, probe_x)
        ylabel = probe_labels.get(probe_y, probe_y)
        layer_str = "Best Layer" if use_best else "Last Layer"
        title = f"{ylabel} vs {xlabel} ({layer_str})"

        layer_suffix = "_best_layer" if use_best else "_last_layer"
        output_path = OUTPUT_DIR / f"probe_vs_probe_{probe_x}_vs_{probe_y}{layer_suffix}"
        plot_probe_vs_probe_grid(merged_by_step, output_path, formats,
                                  xlabel=xlabel, ylabel=ylabel, title=title)
        return

    # Handle scale comparison
    if args.compare_scales:
        use_best = getattr(args, 'best_layer', False)
        merged_by_step = {}
        for step in [0, 1, 2]:
            df = load_probe_metrics_by_scale(probe_db, task=args.task, step=step, use_best_layer=use_best)
            if df.empty:
                merged_by_step[step] = pd.DataFrame()
                continue

            # Match small and large scale for each (model, dataset, bias)
            records = []
            for (model, dataset, bias), group in df.groupby(["model", "dataset", "bias"]):
                if dataset not in SCALE_MAPPING:
                    continue
                small_n = SCALE_MAPPING[dataset]["small"]
                large_n = SCALE_MAPPING[dataset]["large"]
                small_row = group[group["n_questions"] == small_n]
                large_row = group[group["n_questions"] == large_n]
                if not small_row.empty and not large_row.empty:
                    records.append({
                        "model": model,
                        "dataset": dataset,
                        "bias": bias,
                        "small_auc": small_row.iloc[0]["probe_auc"],
                        "large_auc": large_row.iloc[0]["probe_auc"],
                    })
            merged_by_step[step] = pd.DataFrame(records)

        layer_str = "Best Layer" if use_best else "Last Layer"
        task_label = "Motivated vs Others" if args.task == "mot_vs_oth" else args.task
        title = f"{task_label}: Small Scale vs Large Scale ({layer_str})"

        layer_suffix = "_best_layer" if use_best else "_last_layer"
        output_path = OUTPUT_DIR / f"scale_comparison_{args.task}{layer_suffix}"
        plot_scale_comparison_grid(merged_by_step, output_path, formats, title=title)
        return

    # Determine which task to use for LLM metrics
    llm_task = args.llm_task if args.llm_task else args.task

    # Handle grid mode separately
    if args.grid:
        llm_df = load_llm_metrics(llm_db, task=llm_task)
        use_best = getattr(args, 'best_layer', False)

        task_suffix = f"_{args.task}"
        llm_name = args.llm if args.llm else "llm"
        llm_suffix = f"_{llm_name}"
        layer_suffix = "_best_layer" if use_best else ""

        # Use probe_vs_{llm} subfolder for grid plots
        grid_output_dir = Path("figures") / f"probe_vs_{llm_name}"
        grid_output_dir.mkdir(parents=True, exist_ok=True)

        # Determine title based on task
        layer_str = "Best Layer" if use_best else "Last Layer"
        if args.task.startswith("mot_vs_"):
            title = f"Motivated vs Others: Probe ({layer_str}) vs LLM Baseline"
        elif args.task == "has-switched":
            title = f"Motivated vs Aligned: Probe ({layer_str}) vs LLM Baseline"
        else:
            title = f"{args.task}: Probe ({layer_str}) vs LLM Baseline"

        # Build merged data for each step
        all_merged_by_step = {}
        for step in [0, 1, 2]:
            probe_df = load_probe_metrics_by_step(probe_db, task=args.task, step=step, use_best_layer=use_best)
            merged = merge_metrics(probe_df, llm_df)
            if args.llm:
                merged = merged[merged["llm"] == args.llm]
            all_merged_by_step[step] = merged

        # Filter to specific datasets if comma-separated list provided
        dataset_filter_list = None
        dataset_file_suffix = ""
        if args.dataset and args.dataset != "each" and "," in args.dataset:
            dataset_filter_list = [d.strip() for d in args.dataset.split(",")]
            for step in all_merged_by_step:
                all_merged_by_step[step] = all_merged_by_step[step][
                    all_merged_by_step[step]["dataset"].isin(dataset_filter_list)
                ]
            dataset_file_suffix = "_" + "_".join(dataset_filter_list)

        # Determine steps to plot
        step_filter = [args.step] if args.step is not None else [0, 2]
        step_suffix = f"_step{args.step}" if args.step is not None else ""
        step_name = {0: "Preemptive", 2: "Post-Hoc"}.get(args.step, "")

        # Bar chart: 4x3 grid, rows=datasets, cols=models, bars=hint types
        if args.bar:
            probe_by_step = {}
            for step in step_filter:
                probe_df = load_probe_metrics_by_step(probe_db, task=args.task, step=step, use_best_layer=use_best)
                if dataset_filter_list:
                    probe_df = probe_df[probe_df["dataset"].isin(dataset_filter_list)]
                probe_by_step[step] = probe_df

            task_labels = {
                "has-switched": "Motivated vs Aligned",
                "mot_vs_oth": "Motivated vs Others",
            }
            bar_title = f"{task_labels.get(args.task, args.task)}: Probe AUC ({layer_str})"
            if step_name:
                bar_title = f"{bar_title} — {step_name}"
            output_path = grid_output_dir / f"bar{task_suffix}{layer_suffix}{step_suffix}{dataset_file_suffix}"
            plot_bar_grid(probe_by_step, output_path, formats, title=bar_title, steps=step_filter)
            return

        # Average over datasets: 1x3 grid, colored by hint type
        if args.avg_datasets:
            plot_title = ""
            output_path = grid_output_dir / f"probe_vs{llm_suffix}{task_suffix}{layer_suffix}{step_suffix}{dataset_file_suffix}_avg_datasets"
            plot_scatter_avg_over_datasets(all_merged_by_step, output_path, formats, title=plot_title, steps=step_filter)
            return

        # Average over hint types: 1x3 grid, colored by dataset
        if args.avg_biases:
            step_title = {0: "Preemptive", 2: "Post-Hoc"}.get(args.step, "")
            plot_title = f"{step_title} Detection of Motivated Reasoning" if step_title else "Detection of Motivated Reasoning"
            output_path = grid_output_dir / f"probe_vs{llm_suffix}{task_suffix}{layer_suffix}{step_suffix}{dataset_file_suffix}_avg_biases"
            plot_scatter_avg_over_biases(all_merged_by_step, output_path, formats, title=plot_title, steps=step_filter)
            return

        # Combined 3x3 grid with all hint types
        if args.combined_biases:
            all_bias_data = {}
            for bias in BIAS_MARKERS.keys():
                merged_by_step = {}
                for step, merged in all_merged_by_step.items():
                    merged_by_step[step] = merged[merged["bias"] == bias]
                all_bias_data[bias] = merged_by_step

            plot_title = f"{title} — {step_name}" if step_name else title
            output_path = grid_output_dir / f"probe_vs{llm_suffix}{task_suffix}{layer_suffix}{step_suffix}{dataset_file_suffix}_all_biases"
            plot_scatter_grid_all_biases(all_bias_data, output_path, formats, title=plot_title, steps=step_filter)
            return

        # Combined datasets: 4x3 grid, rows = datasets, colored by hint type
        if args.combined_datasets:
            all_dataset_data = {}
            for dataset in DATASET_COLORS.keys():
                merged_by_step = {}
                for step, merged in all_merged_by_step.items():
                    merged_by_step[step] = merged[merged["dataset"] == dataset]
                all_dataset_data[dataset] = merged_by_step

            plot_title = f"{title} — {step_name}" if step_name else title
            output_path = grid_output_dir / f"probe_vs{llm_suffix}{task_suffix}{layer_suffix}{step_suffix}{dataset_file_suffix}_all_datasets"
            plot_scatter_grid_all_datasets(all_dataset_data, output_path, formats, title=plot_title, steps=step_filter)
            return

        # Determine which bias types to plot
        if args.bias == "each":
            bias_list = list(BIAS_MARKERS.keys())
        elif args.bias:
            bias_list = [args.bias]
        else:
            bias_list = [None]  # No filtering

        # Determine which datasets to plot (colored by hint type)
        if args.dataset == "each":
            dataset_list = list(DATASET_COLORS.keys())
        elif args.dataset:
            dataset_list = [args.dataset]
        else:
            dataset_list = [None]  # No filtering

        # If dataset filtering is active, color by hint type
        if dataset_list != [None]:
            for dataset in dataset_list:
                merged_by_step = {}
                for step, merged in all_merged_by_step.items():
                    if dataset is not None:
                        merged_by_step[step] = merged[merged["dataset"] == dataset]
                    else:
                        merged_by_step[step] = merged

                dataset_suffix = f"_{dataset}" if dataset else ""
                dataset_title_label = DATASET_LABELS.get(dataset, dataset) if dataset else ""
                plot_title = f"{title} — {dataset_title_label}" if dataset_title_label else title

                output_path = grid_output_dir / f"probe_vs{llm_suffix}{task_suffix}{layer_suffix}{dataset_suffix}"
                plot_scatter_grid_by_step_color_bias(merged_by_step, output_path, formats, title=plot_title)
            return

        for bias in bias_list:
            merged_by_step = {}
            for step, merged in all_merged_by_step.items():
                if bias is not None:
                    merged_by_step[step] = merged[merged["bias"] == bias]
                else:
                    merged_by_step[step] = merged

            bias_suffix = f"_{bias}" if bias else ""
            bias_title_label = BIAS_LABELS.get(bias, bias) if bias else ""
            plot_title = f"{title} — {bias_title_label}" if bias_title_label else title

            output_path = grid_output_dir / f"probe_vs{llm_suffix}{task_suffix}{layer_suffix}{bias_suffix}"
            plot_scatter_grid_by_step(merged_by_step, output_path, formats, title=plot_title)
        return

    probe_df = load_probe_metrics(probe_db, task=args.task, mode=args.mode)
    llm_df = load_llm_metrics(llm_db, task=llm_task)
    merged = merge_metrics(probe_df, llm_df)

    # Filter by LLM if specified
    if args.llm:
        merged = merged[merged["llm"] == args.llm]

    if merged.empty:
        print(f"No matching data found for task={args.task} (llm_task={llm_task})")
        return

    print(f"\nProbe task: {args.task}")
    print(f"LLM task: {llm_task}")
    print(f"Mode: {args.mode}")
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

    # Add task suffix if not default
    task_suffix = f"_{args.task}" if args.task != "has-switched" else ""

    # Add LLM suffix if filtered
    llm_suffix = f"_{args.llm}" if args.llm else ""

    if args.simple and args.by_model:
        # Simple plots, one per model
        for model in merged["model"].unique():
            model_data = merged[merged["model"] == model]
            model_safe = model.replace(".", "-")
            output_path = OUTPUT_DIR / f"probe_vs_llm_scatter_simple_{model_safe}{task_suffix}{mode_suffix}{llm_suffix}"
            plot_scatter_simple(model_data, output_path, formats)
    elif args.simple:
        output_path = OUTPUT_DIR / f"probe_vs_llm_scatter_simple{task_suffix}{mode_suffix}{llm_suffix}"
        plot_scatter_simple(merged, output_path, formats)
    elif args.by_model:
        # Generate separate plots for each model
        for model in merged["model"].unique():
            # Replace dots in model name to avoid path suffix issues
            model_safe = model.replace(".", "-")
            output_path = OUTPUT_DIR / f"probe_vs_llm_scatter_{model_safe}{task_suffix}{mode_suffix}{llm_suffix}"
            plot_scatter_single_model(merged, model, output_path, formats)
    else:
        output_path = OUTPUT_DIR / f"probe_vs_llm_scatter{task_suffix}{mode_suffix}{llm_suffix}"
        plot_scatter(merged, output_path, formats)


if __name__ == "__main__":
    main()
