#!/usr/bin/env python3
"""
Scatter plot comparing RFM vs Linear probe AUC.

Points above the diagonal mean RFM outperforms Linear; below means Linear wins.
Each model gets its own colour.
"""
import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
})

MOTIVATION_HOME = Path(os.environ.get("MOTIVATION_HOME", "/work/hdd/bbjr/pmirtaheri/motivated"))
OUTPUT_DIR = Path("figures/rfm_vs_linear")
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
    "expert": "o",
    "self": "s",
    "metadata": "^",
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

STEP_LABELS = {0: "Pre-generation", 1: "Mid-generation", 2: "Post-generation"}

TASK_TITLES = {
    "mot_vs_oth": "Motivated vs Other",
    "has-switched": "Post-hoc Detection",
    "general_pregeneration": "General Pre-generation",
    "preemptive": "Preemptive",
}

# Preset aliases: map preset name -> (actual probe task, forced step)
TASK_PRESETS = {
    "general_pregeneration": ("mot_vs_oth", 0),
    "preemptive": ("mot_vs_oth", 0),
}


def load_data(db_path, task, step=None, balanced=0, filter_mentions=1):
    """Load last-layer, last-step AUC for both RFM and Linear classifiers."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT model, dataset, bias, classifier, layer, step, auc
        FROM probe_metrics
        WHERE probe = ? AND classifier IN ('rfm', 'linear')
          AND balanced = ? AND filter_mentions = ?
          AND ckpt_mode = 'rel' AND n_ckpts = 3
    """
    df = pd.read_sql_query(query, conn, params=(task, balanced, filter_mentions))
    conn.close()

    if df.empty:
        return df

    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["layer", "step"])

    records = []
    for (model, dataset, bias, classifier), grp in df.groupby(
        ["model", "dataset", "bias", "classifier"]
    ):
        max_layer = grp["layer"].max()
        layer_df = grp[grp["layer"] == max_layer]
        target_step = step if step is not None else layer_df["step"].max()
        step_df = layer_df[layer_df["step"] == target_step]
        if step_df.empty:
            continue
        row = step_df.iloc[0]
        records.append({
            "model": model,
            "dataset": dataset,
            "bias": bias,
            "classifier": classifier,
            "auc": row["auc"],
        })

    long = pd.DataFrame(records)
    rfm = long[long["classifier"] == "rfm"].rename(columns={"auc": "rfm_auc"}).drop(columns="classifier")
    lin = long[long["classifier"] == "linear"].rename(columns={"auc": "linear_auc"}).drop(columns="classifier")
    merged = rfm.merge(lin, on=["model", "dataset", "bias"])
    return merged


def plot_scatter(df, output_stem, formats, task, step=None):
    """Single scatter: Linear AUC (x) vs RFM AUC (y)."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0.4, 1], [0.4, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    for _, row in df.iterrows():
        ax.scatter(
            row["linear_auc"], row["rfm_auc"],
            color=MODEL_COLORS.get(row["model"], "black"),
            marker="o",
            s=100, edgecolors="white", linewidths=0.5, zorder=3, alpha=0.8,
        )

    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=MODEL_COLORS[m], markersize=10, label=MODEL_LABELS[m])
        for m in MODEL_COLORS if m in df["model"].values
    ]
    ax.legend(handles=model_handles, title="Model", loc="upper left", framealpha=0.9)

    ax.set_xlabel("Linear Probe AUC")
    ax.set_ylabel("RFM Probe AUC")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if task in TASK_TITLES:
        sl = TASK_TITLES[task]
    else:
        sl = STEP_LABELS.get(step if step is not None else 2, f"Step {step}")
    ax.set_title(f"RFM vs Linear \u2014 {sl}")

    fig.tight_layout()
    for fmt in formats:
        out = OUTPUT_DIR / f"{output_stem}.{fmt}"
        dpi = 200 if fmt == "png" else 300
        fig.savefig(out, dpi=dpi)
        print(f"Saved: {out}")
    plt.close(fig)


def plot_combined_scatter(df_pre, df_post, output_stem, formats):
    """Side-by-side scatter: pre-generation (left) and post-generation (right)."""
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax, df, title in [(ax_pre, df_pre, "Pre-generation"), (ax_post, df_post, "Post-generation")]:
        ax.plot([0.5, 1], [0.5, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

        for _, row in df.iterrows():
            ax.scatter(
                row["linear_auc"], row["rfm_auc"],
                color=MODEL_COLORS.get(row["model"], "black"),
                marker="o",
                s=100, edgecolors="white", linewidths=0.5, zorder=3, alpha=0.8,
            )

        ax.set_xlabel("Linear Probe AUC")
        ax.set_xlim(0.5, 1.02)
        ax.set_ylim(0.5, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    ax_pre.set_ylabel("RFM Probe AUC")

    # Single legend
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=MODEL_COLORS[m], markersize=10, label=MODEL_LABELS[m])
        for m in MODEL_COLORS if m in df_pre["model"].values or m in df_post["model"].values
    ]
    ax_post.legend(handles=model_handles, title="Model", loc="upper left", framealpha=0.9)

    fig.tight_layout()
    for fmt in formats:
        out = OUTPUT_DIR / f"{output_stem}.{fmt}"
        dpi = 200 if fmt == "png" else 300
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_grouped_bar(df, output_stem, formats, task, step=None):
    """Grouped bar chart: side-by-side RFM vs Linear AUC per condition."""
    df = df.copy()
    df = df.sort_values(["model", "dataset", "bias"]).reset_index(drop=True)

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(16, len(df) * 0.9), 6))
    ax.bar([i - width / 2 for i in x], df["rfm_auc"], width,
           label="RFM", color="#2c7fb8", edgecolor="white", linewidth=0.5)
    ax.bar([i + width / 2 for i in x], df["linear_auc"], width,
           label="Linear", color="#a1dab4", edgecolor="white", linewidth=0.5)

    tick_labels = []
    for _, row in df.iterrows():
        ds = DATASET_ABBREV.get(row["dataset"], row["dataset"][:3])
        bias = BIAS_LABELS.get(row["bias"], row["bias"])[:4]
        tick_labels.append(f"{ds}/{bias}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=11)

    models = df["model"].unique()
    for model in models:
        idxs = df.index[df["model"] == model].tolist()
        mid = (idxs[0] + idxs[-1]) / 2
        ax.annotate(MODEL_LABELS.get(model, model),
                    xy=(mid, 0.38), fontsize=11, ha="center", fontweight="bold",
                    color=MODEL_COLORS.get(model, "black"))

    for i in range(1, len(models)):
        boundary = df.index[df["model"] == models[i]].min() - 0.5
        ax.axvline(boundary, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_ylabel("AUC")
    ax.set_ylim(0.35, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    if task in TASK_TITLES:
        sl = TASK_TITLES[task]
    else:
        sl = STEP_LABELS.get(step if step is not None else 2, f"Step {step}")
    ax.set_title(f"RFM vs Linear \u2014 {sl} (Last Layer)")

    fig.tight_layout()
    for fmt in formats:
        out = OUTPUT_DIR / f"{output_stem}_bar.{fmt}"
        dpi = 200 if fmt == "png" else 300
        fig.savefig(out, dpi=dpi)
        print(f"Saved: {out}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RFM vs Linear probe AUC.")
    parser.add_argument("--fmt", action="append", default=None,
                        help="Output formats (default: png).")
    parser.add_argument("--task", default="mot_vs_alg",
                        help="Probe task (default: mot_vs_alg).")
    parser.add_argument("--step", type=int, default=None,
                        help="CoT step to use (0=pre, 1=mid, 2=post). Default: last step.")
    parser.add_argument("--combined", action="store_true",
                        help="Generate combined pre+post scatter for the given task.")
    return parser.parse_args()


def main():
    args = parse_args()
    formats = args.fmt or ["png"]
    db_path = MOTIVATION_HOME / "probe_metrics.db"
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    # Resolve preset aliases
    task_name = args.task
    if task_name in TASK_PRESETS:
        probe_task, preset_step = TASK_PRESETS[task_name]
        step = preset_step if args.step is None else args.step
    else:
        probe_task = task_name
        step = args.step

    if args.combined:
        df_pre = load_data(db_path, task=probe_task, step=0)
        df_post = load_data(db_path, task=probe_task, step=None)
        if df_pre.empty or df_post.empty:
            print("No data found for combined plot.")
            return
        print(f"Combined: {len(df_pre)} pre-gen, {len(df_post)} post-gen conditions")
        stem = f"rfm_vs_linear_{task_name}_combined"
        plot_combined_scatter(df_pre, df_post, stem, formats)
        return

    df = load_data(db_path, task=probe_task, step=step)
    if df.empty:
        print("No data found.")
        return

    print(f"Loaded {len(df)} conditions for task={task_name} (probe={probe_task}, step={step})")
    print(df.to_string(index=False))

    step_label = {0: "pre_generation", 1: "mid_generation", 2: "post_generation"}.get(
        step if step is not None else 2, f"step{step}")
    stem = f"rfm_vs_linear_{task_name}_{step_label}"
    plot_scatter(df, stem, formats, task_name, step)
    plot_grouped_bar(df, stem, formats, task_name, step)


if __name__ == "__main__":
    main()
