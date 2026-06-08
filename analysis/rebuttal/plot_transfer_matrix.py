"""Cross-bias transfer matrix at Gemma-3-27B (ARC-Challenge, RFM, layer 62).

Two panels (pre-CoT, post-CoT). Each panel is a 2x2 matrix where rows are the
training bias and columns are the evaluation bias. Cell text is the raw AUC
(direction-aware), but cell *color* is the discrimination score
max(AUC, 1-AUC), so anti-aligned cells (AUC < 0.5) are colored by how far
they are from chance, not by their raw value.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BIAS_LABELS = {"self": "Consistency", "metadata": "Metadata"}
BIAS_ORDER = ["self", "metadata"]

# Pulled from probe_metrics.db (model=gemma-3-27b, dataset=arc-challenge,
# probe=mot_vs_alg, classifier=rfm, layer=62, filter_mentions=1, balanced=0).
# Same-bias diagonals: tag=''. Off-diagonals: tag='eval_bias=<other>'.
AUCS = {
    "pre": {  # step 0
        ("self", "self"):         0.9904,
        ("self", "metadata"):     0.6279,
        ("metadata", "self"):     0.1764,
        ("metadata", "metadata"): 0.9408,
    },
    "post": {  # step 2
        ("self", "self"):         0.9835,
        ("self", "metadata"):     0.9312,
        ("metadata", "self"):     0.9167,
        ("metadata", "metadata"): 0.9858,
    },
}

PANEL_TITLES = {"pre": "Pre-CoT", "post": "Post-CoT"}


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


def discrim(auc):
    return max(auc, 1.0 - auc)


def draw_panel(ax, aucs, title, vmin=0.5, vmax=1.0, cmap_name="YlGn"):
    n = len(BIAS_ORDER)
    color_mat = np.zeros((n, n))
    for i, train_bias in enumerate(BIAS_ORDER):
        for j, eval_bias in enumerate(BIAS_ORDER):
            color_mat[i, j] = discrim(aucs[(train_bias, eval_bias)])
    im = ax.imshow(color_mat, cmap=cmap_name, vmin=vmin, vmax=vmax,
                   aspect="equal", origin="lower")

    for i, train_bias in enumerate(BIAS_ORDER):
        for j, eval_bias in enumerate(BIAS_ORDER):
            auc = aucs[(train_bias, eval_bias)]
            d = discrim(auc)
            # White text on darker cells, black on lighter ones.
            # Threshold ~0.78 keeps text readable across YlGn at this range.
            txt_color = "white" if d >= 0.78 else "black"
            ax.text(j, i, f"{auc:.2f}", ha="center", va="center",
                    fontsize=14, color=txt_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([BIAS_LABELS[b] for b in BIAS_ORDER])
    ax.set_yticklabels([BIAS_LABELS[b] for b in BIAS_ORDER])
    ax.set_xlabel("Eval bias")
    if title.startswith("Pre"):
        ax.set_ylabel("Train bias")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmap", default="YlGn")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6),
                              gridspec_kw={"wspace": 0.35})

    last_im = None
    for ax, key in zip(axes, ["pre", "post"]):
        last_im = draw_panel(ax, AUCS[key], PANEL_TITLES[key], cmap_name=args.cmap)

    # Hide y-tick labels on the right panel so they don't overlap the left
    # panel's cell text.
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.04, pad=0.04,
                         ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.set_label("Discrimination = max(AUC, 1$-$AUC)")
    cbar.outline.set_visible(False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
