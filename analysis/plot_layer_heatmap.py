#!/usr/bin/env python3
"""
Layer-wise heatmap of probe AUC averaged over datasets and hint types.

Produces a 1×2 figure:
  Left:  pre-generation  (step=0)
  Right: post-generation (step=2)
Each panel: layers (y-axis) × models (x-axis), AUC averaged over all
datasets and hint types.
"""
import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
})

MOTIVATION_HOME = Path(os.environ.get(
    "MOTIVATION_HOME", "/work/hdd/bbjr/pmirtaheri/motivated"))
DB_PATH = MOTIVATION_HOME / "probe_metrics.db"

MODEL_ORDER = ["qwen-3-8b", "llama-3.1-8b", "gemma-3-4b"]
MODEL_DISPLAY = {
    "qwen-3-8b": "Qwen-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gemma-3-4b": "Gemma-3-4B",
}

STEP_LABELS = {0: "Pre-generation", 2: "Post-generation"}

# 10 evenly spaced layers per model (the base set before adding first/last 3)
EVENLY_SPACED = {
    "qwen-3-8b":    [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
    "llama-3.1-8b": [0, 4, 7, 11, 14, 18, 21, 25, 28, 32],
    "gemma-3-4b":   [0, 4, 8, 12, 15, 19, 23, 26, 30, 34],
}
MAX_LAYER = {"qwen-3-8b": 36, "llama-3.1-8b": 32, "gemma-3-4b": 34}


def load_data(probe: str = "mot_vs_alg", classifier: str = "rfm",
              n_ckpts: int = 3) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT model, dataset, bias, layer, step, auc
        FROM   probe_metrics
        WHERE  probe      = ?
           AND classifier = ?
           AND n_ckpts    = ?
           AND ckpt_mode  = 'rel'
        ORDER BY model, dataset, bias, layer, step
    """
    df = pd.read_sql_query(query, conn, params=(probe, classifier, n_ckpts))
    conn.close()
    return df


def make_heatmap_data(df: pd.DataFrame, step: int):
    """Return matrix and labels using only the 10 evenly spaced layers
    (excluding layer 0 → 9 rows), with normalized layer position labels."""
    sub = df[(df["step"] == step)].copy()

    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]

    model_data = {}
    for model in models:
        keep = [l for l in EVENLY_SPACED[model] if l != 0]
        msub = sub[(sub["model"] == model) & (sub["layer"].isin(keep))]
        agg = msub.groupby("layer")["auc"].mean().sort_index()
        model_data[model] = agg.values

    # Normalized labels: layer / max_layer (same for all models since evenly spaced)
    ref_model = models[0]
    ref_layers = [l for l in EVENLY_SPACED[ref_model] if l != 0]
    norm_labels = [f"{l / MAX_LAYER[ref_model]:.2f}" for l in ref_layers]

    return models, model_data, norm_labels


def plot_figure(df: pd.DataFrame, output_path: Path, formats: list[str]):
    steps = [0, 2]
    vmin, vmax = 0.5, 1.0

    fig, axes = plt.subplots(
        len(steps), 1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={"hspace": 0.45, "right": 0.92, "top": 0.92,
                      "bottom": 0.12, "left": 0.15},
    )

    ims = []
    for row, step in enumerate(steps):
        ax = axes[row]
        models, model_data, norm_labels = make_heatmap_data(df, step)

        # Build matrix: rows = models, cols = layer index
        mat = np.row_stack([model_data[m] for m in models])

        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ims.append(im)

        # Y-axis: model names
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in models])

        # X-axis: only label first and last columns
        ax.set_xticks([0, mat.shape[1] - 1])
        ax.set_xticklabels(["First", "Last"])

        if row == len(steps) - 1:
            ax.set_xlabel("Layer")

        ax.set_title(STEP_LABELS[step], fontsize=20)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.72])
    fig.colorbar(ims[0], cax=cbar_ax, label="AUC")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt == "png" else 300
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot layer-wise heatmaps for probe detection.")
    p.add_argument("--probe", default="mot_vs_alg")
    p.add_argument("--classifier", default="rfm", choices=["rfm", "linear"])
    p.add_argument("--fmt", action="append", default=None,
                   help="Output formats (e.g. --fmt png --fmt pdf).")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    formats = args.fmt or ["png"]

    df = load_data(args.probe, args.classifier)
    if df.empty:
        print("No data found.")
        return

    print(f"Loaded {len(df)} rows")

    out_dir = args.output_dir or (
        MOTIVATION_HOME / "overleaf" / "arxiv" / "figures_ext")
    suffix = f"_{args.classifier}" if args.classifier != "rfm" else ""
    out_path = out_dir / f"layer_heatmap_{args.probe}{suffix}"

    plot_figure(df, out_path, formats)


if __name__ == "__main__":
    main()
