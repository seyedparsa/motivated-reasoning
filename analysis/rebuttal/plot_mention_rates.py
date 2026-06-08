"""Rebuttal Figure: per-(hint type, response category) hint-mention rate by model.

For each model, bias type, and response category, computes:

    mention_rate = mention_count / (mention_count + no_mention_count)

i.e. the fraction of responses *within that category* whose CoT mentions the
hint keyword. One panel per hint type; bars grouped by category on the x-axis,
two bars per group (one per model).

Reads `outputs/taxonomy_metrics/taxonomy_{model}_{dataset}.csv` files.

Usage:
    python analysis/rebuttal/plot_mention_rates.py \\
        --models gemma-3-4b gemma-3-27b \\
        --dataset arc-challenge \\
        --out figures/rebuttal/mention_rates_gemma_arc-challenge.pdf
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CATEGORIES = ["motivated", "resistant", "aligned", "other"]
CATEGORY_LABELS = {
    "motivated": "Motivated",
    "resistant": "Resistant",
    "aligned": "Aligned",
    "other": "Other",
}
CATEGORY_COLORS = {
    "motivated": "#FF6B6B",
    "resistant": "#51CF66",
    "aligned": "#4DABF7",
    "other": "#B197FC",
}

HINT_TYPES = [
    ("expert", "Sycophancy"),
    ("metadata", "Metadata"),
    ("self", "Consistency"),
]
# Note: the keyword check ('hint'/'expert'/'metadata') is structurally blind to
# Consistency (the bias never introduces those words into the prompt), so the
# Consistency panel is essentially a near-zero floor — included here as a
# visual reminder that keyword-based monitoring fails on this bias type.

# Display labels for models (legend / titles use these).
MODEL_LABELS = {
    "gemma-3-4b": "Gemma-3-4B",
    "gemma-3-27b": "Gemma-3-27B",
    "gemma-3-4b-it": "Gemma-3-4B",
    "gemma-3-27b-it": "Gemma-3-27B",
}

# Lightness applied to the category color for the smaller (first) model.
SMALL_LIGHTEN = 0.55


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def lighten(color, amount=SMALL_LIGHTEN):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def load_mention_rates(csv_path: Path, bias: str) -> dict:
    """For one bias type, return {category: (mention_rate_pct, n_total)}.

    n_total is mention_count + no_mention_count (responses in that category).
    Categories outside CATEGORIES (departing/shifting/invalid) are folded into
    'other'. Returns NaN rate when n_total == 0.
    """
    df = pd.read_csv(csv_path)
    sub = df[(df["bias_type"] == bias) & (df["hint_choice"] == "ALL")]

    def counts_for_subset(subset_name):
        rows = sub[sub["subset"] == subset_name]
        c = {cat: 0 for cat in CATEGORIES}
        for _, row in rows.iterrows():
            cat = row["category"]
            n = int(row["count"])
            if cat in c:
                c[cat] += n
            else:
                c["other"] += n
        return c

    m = counts_for_subset("mention")
    nm = counts_for_subset("no_mention")

    out = {}
    for cat in CATEGORIES:
        total = m[cat] + nm[cat]
        rate = (100.0 * m[cat] / total) if total > 0 else float("nan")
        out[cat] = (rate, total)
    return out


def draw_panel(ax, per_model_rates, models, title):
    """per_model_rates: list (one entry per model) of {category: (rate_pct, n)}.

    For each category, draws one bar per model side-by-side. Bar color is the
    category color. Models are distinguished by lightness (smaller -> lighter,
    larger -> full saturation).
    """
    n_cats = len(CATEGORIES)
    n_models = len(models)
    bar_w = 0.8 / n_models
    x = np.arange(n_cats)

    for mi, model in enumerate(models):
        rates = per_model_rates[mi]
        offsets = (mi - (n_models - 1) / 2) * bar_w
        for ci, cat in enumerate(CATEGORIES):
            rate, n = rates[cat]
            color = CATEGORY_COLORS[cat]
            face = lighten(color) if mi == 0 else color
            if np.isnan(rate):
                continue
            ax.bar(
                x[ci] + offsets, rate, bar_w,
                color=face, edgecolor="black", linewidth=0.6,
            )
            ax.text(
                x[ci] + offsets, rate + 1.5, f"{rate:.0f}%",
                ha="center", va="bottom", fontsize=9, color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Hint verbalization rate (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Models in plot order (left=lighter, right=full saturation)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--csv-dir", default="outputs/taxonomy_metrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    per_model_per_bias = {}
    for m in args.models:
        path = csv_dir / f"taxonomy_{m}_{args.dataset}.csv"
        if not path.exists():
            raise SystemExit(f"missing CSV: {path}")
        per_model_per_bias[m] = {bias: load_mention_rates(path, bias) for bias, _ in HINT_TYPES}

    fig, axes = plt.subplots(1, len(HINT_TYPES),
                              figsize=(3.6 * len(HINT_TYPES), 3.4),
                              sharey=True)
    if len(HINT_TYPES) == 1:
        axes = [axes]

    for ax, (bias, label) in zip(axes, HINT_TYPES):
        per_model_rates = [per_model_per_bias[m][bias] for m in args.models]
        draw_panel(ax, per_model_rates, args.models, label)

    # Only the leftmost panel keeps the y-axis label (others share via sharey).
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Legend at top center, single row, matching Figure 4 of the paper.
    ref = CATEGORY_COLORS["motivated"]
    handles = [
        Patch(facecolor=lighten(ref), edgecolor="black",
              label=MODEL_LABELS.get(args.models[0], args.models[0])),
        Patch(facecolor=ref, edgecolor="black",
              label=MODEL_LABELS.get(args.models[1], args.models[1])),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    if args.title:
        fig.suptitle(args.title, fontsize=12, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
