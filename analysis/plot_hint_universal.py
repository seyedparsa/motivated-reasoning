#!/usr/bin/env python3
"""
3-panel heatmap: hint-recovery probe accuracy across layers and CoT steps.

Left:   prefix[k] mode  — steps from the beginning of CoT
Middle: rel mode         — normalised steps (0 … 1)
Right:  suffix[k] mode  — steps from the end of CoT

Red curved arrows + dashed box connect the left/right panels to the
corresponding edges of the normalised (middle) panel.
"""
import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 34,
    "axes.titlesize": 38,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 30,
    "legend.title_fontsize": 34,
})

MOTIVATION_HOME = Path(os.environ.get(
    "MOTIVATION_HOME", "/work/hdd/bbjr/pmirtaheri/motivated"))
DB_PATH = MOTIVATION_HOME / "probe_metrics.db"


# ── data loading ───────────────────────────────────────────────────────
def load_data(model: str, dataset: str, bias: str,
              n_ckpts: int, classifier: str = "rfm") -> pd.DataFrame:
    """Load h_recovery probe metrics for the 3 ckpt modes."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT ckpt_mode, layer, step, accuracy, auc
        FROM   probe_metrics
        WHERE  probe       = 'h_recovery'
           AND model       = ?
           AND dataset     = ?
           AND bias        = ?
           AND n_ckpts     = ?
           AND classifier  = ?
        ORDER BY ckpt_mode, layer, step
    """
    df = pd.read_sql_query(query, conn,
                           params=(model, dataset, bias, n_ckpts, classifier))
    conn.close()
    return df


# ── x-axis label helpers ───────────────────────────────────────────────
def prefix_labels(steps, stride: int):
    """0, 5, 10, 15, 20  (token offset from generation start)."""
    return [int(s * stride) for s in steps]


def rel_labels(steps, n_ckpts: int):
    """0.00, 0.25, 0.50, 0.75, 1.00  (normalised position)."""
    return [f"{s / (n_ckpts - 1):.2f}" for s in steps]


def suffix_labels(steps, n_ckpts: int, stride: int):
    """20, 15, 10, 5, 0  (steps to the end of CoT)."""
    return [int((n_ckpts - 1 - s) * stride) for s in steps]


# ── single-panel heatmap ──────────────────────────────────────────────
def _plot_panel(ax, pivot, vmin, vmax, annotate=False, annotate_layers=None):
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, origin="lower")
    if annotate:
        mid = (vmin + vmax) / 2
        layers = list(pivot.index)
        for i in range(pivot.shape[0]):
            if annotate_layers is not None and layers[i] not in annotate_layers:
                continue
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val < mid else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=24)
    return im


# ── main figure ────────────────────────────────────────────────────────
def plot_figure(df: pd.DataFrame, n_ckpts: int, stride: int,
                output_path: Path, formats: list[str],
                metric: str = "accuracy", annotate: bool = False,
                annotate_layers: list[int] | None = None):
    # Separate by ckpt_mode
    prefix_key = f"prefix[{stride}]"
    suffix_key = f"suffix[{stride}]"
    rel_key = "rel"

    modes = {prefix_key: "prefix", rel_key: "rel", suffix_key: "suffix"}
    panels = {}
    for mode_val, label in modes.items():
        sub = df[df["ckpt_mode"] == mode_val].copy()
        if sub.empty:
            print(f"Warning: no data for ckpt_mode={mode_val}")
            continue
        sub["layer"] = sub["layer"].astype(int)
        sub["step"] = sub["step"].astype(int)
        sub = sub[sub["layer"] != 0]
        pivot = sub.pivot_table(index="layer", columns="step",
                                values=metric, aggfunc="first")
        pivot = pivot.sort_index()  # layers ascending (origin="lower")
        panels[label] = pivot

    if len(panels) < 3:
        missing = set(modes.values()) - set(panels.keys())
        print(f"Missing panels: {missing}. Available ckpt_modes in DB: "
              f"{sorted(df['ckpt_mode'].unique())}")
        if not panels:
            return
        # proceed with what we have

    panel_order = ["prefix", "rel", "suffix"]
    panel_order = [p for p in panel_order if p in panels]

    # global colour range
    if metric == "accuracy":
        # accuracy is stored as percentage (0–100), convert to fraction
        for k in panels:
            panels[k] = panels[k] / 100.0
    vmin = min(p.min().min() for p in panels.values())
    vmax = max(p.max().max() for p in panels.values())
    vmin = 0.25
    vmax = 1.0

    fig, axes = plt.subplots(
        1, len(panel_order), figsize=(8 * len(panel_order) + 3, 14),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "right": 0.88, "top": 0.90}
    )
    if len(panel_order) == 1:
        axes = [axes]

    ims = []
    for ax, pname in zip(axes, panel_order):
        pivot = panels[pname]
        im = _plot_panel(ax, pivot, vmin, vmax, annotate=annotate,
                         annotate_layers=annotate_layers)
        ims.append(im)

        steps = sorted(pivot.columns)
        layers = sorted(pivot.index)

        # x-axis labels
        ax.set_xticks(range(len(steps)))
        if pname == "prefix":
            ax.set_xticklabels(prefix_labels(steps, stride))
            ax.set_xlabel("Tokens from Beginning")
        elif pname == "rel":
            ax.set_xticklabels(rel_labels(steps, n_ckpts))
            ax.set_xlabel("Normalized Token Index")
        elif pname == "suffix":
            ax.set_xticklabels(suffix_labels(steps, n_ckpts, stride))
            ax.set_xlabel("Tokens to End")

        # y-axis
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers])

    axes[0].set_ylabel("Layer", labelpad=12)

    # shared colour bar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    cbar_label = "Accuracy" if metric == "accuracy" else "AUC"
    fig.colorbar(ims[0], cax=cbar_ax, label=cbar_label)

    # ── red annotation arrows (prefix ↔ middle ← → suffix) ────────
    if len(panel_order) == 3:
        _draw_arrows(fig, axes, panel_order)

    # save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = output_path.with_suffix(f".{fmt}")
        dpi = 200 if fmt == "png" else 300
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def _draw_arrows(fig, axes, panel_order):
    """Draw red brackets on top of the prefix/suffix panels, with
    curved arrows from the bracket start to the corresponding dashed
    box on the normalised (middle) panel."""
    ax_prefix = axes[panel_order.index("prefix")]
    ax_rel = axes[panel_order.index("rel")]
    ax_suffix = axes[panel_order.index("suffix")]

    r_bbox = ax_rel.get_position()
    p_bbox = ax_prefix.get_position()
    s_bbox = ax_suffix.get_position()

    # Width of each dashed box (fraction of middle panel)
    box_frac = 0.18
    box_w = r_bbox.width * box_frac
    pad = 0.005
    x_inset = 0.01   # shift boxes inward to avoid tick-label overlap
    y_inset = 0.012  # raise boxes to avoid x-tick label overlap

    # Left dashed box (beginning edge of normalised panel)
    rect_left = mpatches.FancyBboxPatch(
        (r_bbox.x0 - pad + x_inset, r_bbox.y0 - pad + y_inset),
        box_w + 2 * pad, r_bbox.height - y_inset,
        boxstyle="round,pad=0.005",
        linewidth=5, edgecolor="red", facecolor="none",
        linestyle="--", transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_left)

    # Right dashed box (end edge of normalised panel)
    rect_right = mpatches.FancyBboxPatch(
        (r_bbox.x1 - box_w - pad - x_inset, r_bbox.y0 - pad + y_inset),
        box_w + 2 * pad, r_bbox.height - y_inset,
        boxstyle="round,pad=0.005",
        linewidth=5, edgecolor="red", facecolor="none",
        linestyle="--", transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_right)



# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot 3-panel hint-recovery heatmap (Figure 4).")
    p.add_argument("--model", default="qwen-3-8b")
    p.add_argument("--dataset", default="mmlu")
    p.add_argument("--bias", default="expert")
    p.add_argument("--n-ckpts", type=int, default=5)
    p.add_argument("--stride", type=int, default=5,
                   help="Stride used in prefix[k]/suffix[k] modes.")
    p.add_argument("--classifier", default="rfm",
                   choices=["rfm", "linear"])
    p.add_argument("--metric", default="accuracy",
                   choices=["accuracy", "auc"])
    p.add_argument("--annotate", action="store_true",
                   help="Write values on heatmap cells.")
    p.add_argument("--annotate-layers", type=int, nargs="+", default=None,
                   help="Only annotate these layers (e.g. --annotate-layers 20).")
    p.add_argument("--fmt", action="append", default=None,
                   help="Output formats (e.g. --fmt png --fmt pdf).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output directory.")
    return p.parse_args()


def main():
    args = parse_args()
    formats = args.fmt or ["png"]

    df = load_data(args.model, args.dataset, args.bias,
                   args.n_ckpts, args.classifier)
    if df.empty:
        print(f"No data found for {args.model}/{args.dataset}/{args.bias} "
              f"n_ckpts={args.n_ckpts} classifier={args.classifier}")
        return

    print(f"Loaded {len(df)} rows  |  ckpt_modes: "
          f"{sorted(df['ckpt_mode'].unique())}")

    out_dir = args.output_dir or (
        MOTIVATION_HOME / "overleaf" / "arxiv" / "figures_ext")
    suffix = f"_{args.classifier}" if args.classifier != "rfm" else ""
    out_path = out_dir / f"hint_universal_all_data_box{suffix}"

    plot_figure(df, args.n_ckpts, args.stride, out_path, formats,
                metric=args.metric, annotate=args.annotate,
                annotate_layers=args.annotate_layers)


if __name__ == "__main__":
    main()
