"""Rebuttal Figure: Gemma-3 scale comparison of probe vs CoT monitor AUC.

Mirrors Figure 1 (pre-generation) and Figure 3 (post-generation) of the paper,
but replaces the (Qwen, Llama, Gemma-4B) columns with (Gemma-4B, Gemma-27B).
Each panel is a scatter where the x-axis is the GPT-5-nano CoT monitor AUC and
the y-axis is the RFM probe AUC; one point per hint type.

Data is loaded from sqlite metric DBs that live on the delta cluster, so this
script prefers a `--values-json` override containing pre-pulled numbers when
the DBs aren't reachable locally. Run:

    python analysis/rebuttal/plot_probe_vs_monitor.py \\
        --step 0 --out figures/rebuttal/probe_vs_monitor_pre.pdf

with --step 0 (pre-CoT, Figure 1) or --step 2 (post-CoT, Figure 3).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HINT_TYPES = [
    ("self", "Consistency"),
    ("metadata", "Metadata"),
]
# Sycophancy is omitted: 27B's no-mention test pool collapses to ~30 examples
# (the model verbalizes the expert hint nearly always), so it can't be
# compared against 4B on equal footing.
BIAS_COLORS = {
    "expert": "#1b9e77",
    "self": "#d95f02",
    "metadata": "#7570b3",
}
BIAS_LABELS = {b: lbl for b, lbl in HINT_TYPES}

MODEL_LABELS = {
    "gemma-3-4b": "Gemma-3-4B-it",
    "gemma-3-27b": "Gemma-3-27B-it",
}

# Numbers compiled from probe_metrics.db (RFM, last layer per model: 4B=34,
# 27B=62; filter_mentions=1, balanced=0, tag='') and llm_metrics.db
# (gpt-5-nano CoT monitor on the no-mention slice). 27B/expert is omitted
# because the test pool collapses to ~30 examples.
DEFAULT_VALUES = {
    "gemma-3-4b": {
        "expert":   {"monitor": 0.7926, "pre": 0.5475, "post": 0.7442},
        "self":     {"monitor": 0.8419, "pre": 0.9322, "post": 0.8843},
        "metadata": {"monitor": 0.7996, "pre": 0.6989, "post": 0.8723},
    },
    "gemma-3-27b": {
        # expert intentionally omitted (n=30 in test pool).
        "self":     {"monitor": 0.7424, "pre": 0.9904, "post": 0.9835},
        "metadata": {"monitor": 0.7242, "pre": 0.9408, "post": 0.9858},
    },
}


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def draw_panel(ax, points, model_label, axis_min, axis_max, step):
    """points: list of (bias_key, monitor_auc, probe_auc)."""
    # Diagonal reference line.
    ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--", alpha=0.45, linewidth=1.3)

    # Green-shaded region where probe > monitor.
    ax.fill_between(
        [axis_min, axis_max], [axis_min, axis_max], [axis_max, axis_max],
        alpha=0.10, color="green",
    )

    # Direction-of-better-than annotations.
    arrow_kw = dict(arrowstyle="->", color="gray", lw=1.2)
    ax.annotate("", xy=(0.72, 0.555), xytext=(0.605, 0.555), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.555, 0.72), xytext=(0.555, 0.605), arrowprops=arrow_kw)
    if step == 0:
        ax.text(0.605, 0.55, "CoT Monitor Better\n(Needs CoT)",
                ha="left", va="top", fontsize=8, color="gray")
        ax.text(0.55, 0.605, "Probe Better\n(No CoT Generation)",
                ha="right", va="bottom", fontsize=8, color="gray", rotation=90)
    else:
        ax.text(0.605, 0.55, "CoT Monitor Better",
                ha="left", va="top", fontsize=8, color="gray")
        ax.text(0.55, 0.605, "Probe Better",
                ha="right", va="bottom", fontsize=8, color="gray", rotation=90)

    for bias, x, y in points:
        ax.scatter(
            x, y, c=BIAS_COLORS[bias], marker="o",
            s=140, alpha=0.92, edgecolors="black", linewidth=0.7,
            zorder=5,
        )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(model_label)
    ax.set_xlabel("CoT Monitor AUC")


def load_values(args):
    if args.values_json:
        with open(args.values_json) as f:
            return json.load(f)
    return DEFAULT_VALUES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[0, 2], required=True,
                    help="0 = pre-CoT (Figure 1 equivalent), "
                         "2 = post-CoT (Figure 3 equivalent)")
    ap.add_argument("--models", nargs="+", default=["gemma-3-4b", "gemma-3-27b"],
                    help="Models to plot, in left-to-right order")
    ap.add_argument("--values-json", default=None,
                    help="Optional JSON override with the same shape as DEFAULT_VALUES")
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis-min", type=float, default=0.5)
    ap.add_argument("--axis-max", type=float, default=1.0)
    args = ap.parse_args()

    values = load_values(args)
    step_key = "pre" if args.step == 0 else "post"

    n_panels = len(args.models)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.8),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, model in zip(axes, args.models):
        if model not in values:
            raise SystemExit(f"missing values for model {model!r}")
        points = []
        for bias, _ in HINT_TYPES:
            d = values[model].get(bias)
            if d is None:
                continue
            points.append((bias, d["monitor"], d[step_key]))
        draw_panel(ax, points, MODEL_LABELS.get(model, model),
                   args.axis_min, args.axis_max, args.step)

    axes[0].set_ylabel("Probe AUC")

    # Legend at the bottom: hint types only, single row.
    handles = [
        Patch(facecolor=BIAS_COLORS[b], edgecolor="black", label=BIAS_LABELS[b])
        for b, _ in HINT_TYPES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
