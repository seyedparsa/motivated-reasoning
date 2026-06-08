"""Rebuttal Figure: compare response category distributions across model sizes
on a single dataset, for each hint type plus an average panel.

Reads `outputs/taxonomy_metrics/taxonomy_{model}_{dataset}.csv` files (produced
by `main.py --evaluate`) and renders one horizontal stacked bar per model per
hint type. The "average" panel averages the percentage of each category across
the three hint types.

Style mirrors `analysis/plot_categories.py` (same colors / category set), but
the layout is comparison-first: rows = models, columns = hint types + average.

Usage:
    python analysis/rebuttal/plot_size_comparison.py \\
        --models gemma-3-4b gemma-3-27b \\
        --dataset arc-challenge \\
        --out figures/rebuttal/size_comparison_gemma_arc-challenge.pdf
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CATEGORIES = ["motivated", "resistant", "aligned", "other"]
CATEGORY_COLORS = {
    "motivated": "#FF6B6B",
    "resistant": "#51CF66",
    "aligned": "#4DABF7",
    "other": "#B197FC",
}

# Amount to blend toward white for the "mentions hint" sub-segment.
# 0.0 = original color, 1.0 = pure white.
MENTION_LIGHTEN = 0.55


def lighten(color, amount=MENTION_LIGHTEN):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

# Hint columns to render and their display labels (paper terminology)
HINT_TYPES = [
    ("expert", "Sycophancy"),
    ("self", "Consistency"),
    ("metadata", "Metadata"),
]

# Display labels for models (legend / y-tick labels use these).
MODEL_LABELS = {
    "gemma-3-4b": "Gemma-3-4B",
    "gemma-3-27b": "Gemma-3-27B",
    "gemma-3-4b-it": "Gemma-3-4B",
    "gemma-3-27b-it": "Gemma-3-27B",
}

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


def _resolve_condition(condition: str) -> set:
    """Parse a condition spec into the set of categories to drop from the denominator.

    Accepts either a single keyword ('all', 'non_aligned', 'non_resistant',
    'non_other', 'mot_vs_alg') or a comma-separated combination
    (e.g. 'non_aligned,non_resistant') that stacks multiple exclusions.
    """
    if condition == "all":
        return set()
    drops = set()
    for tok in condition.split(","):
        tok = tok.strip()
        if tok == "non_aligned":
            drops.add("aligned")
        elif tok == "non_resistant":
            drops.add("resistant")
        elif tok == "non_other":
            drops.add("other")
        elif tok == "mot_vs_alg":
            # Convenience: keep only motivated + aligned bands.
            drops.update({"resistant", "other"})
        elif tok == "all":
            continue
        else:
            raise ValueError(f"unknown condition token: {tok!r}")
    return drops


def load_overall_counts_and_percents(csv_path: Path, bias: str, subset: str = "overall",
                                      condition: str = "all") -> tuple:
    """Return ({category: (percent, count)}, total_n) for the given subset/bias.

    Same semantics as load_overall_percents but also surfaces raw counts so the
    plotter can annotate absolute numbers alongside percentages.
    """
    df = pd.read_csv(csv_path)
    sub = df[(df["subset"] == subset) & (df["bias_type"] == bias) & (df["hint_choice"] == "ALL")]
    if sub.empty:
        raise ValueError(f"no {subset} rows for bias={bias} in {csv_path}")

    counts = {c: 0 for c in CATEGORIES}
    for _, row in sub.iterrows():
        cat = row["category"]
        n = int(row["count"])
        if cat in counts:
            counts[cat] += n
        else:
            counts["other"] += n

    drops = _resolve_condition(condition)
    denom = sum(counts[c] for c in CATEGORIES if c not in drops)
    for c in drops:
        counts[c] = 0

    if denom == 0:
        return ({c: (0.0, 0) for c in CATEGORIES}, 0)
    return ({c: (100.0 * counts[c] / denom, counts[c]) for c in CATEGORIES}, denom)


def load_overall_percents(csv_path: Path, bias: str, subset: str = "overall",
                          condition: str = "all") -> dict:
    """Return {category: percent} for the given subset of one bias_type.

    subset:    which mention slice to use
      'overall'    -> all hinted responses
      'no_mention' -> CoT does NOT mention hint keyword
      'mention'    -> CoT mentions hint keyword

    condition: which categories form the denominator (renormalization)
      'all'         -> all categories sum to 100% (no filter)
      'non_aligned' -> drop 'aligned' from the denominator, renormalize over
                       the remaining categories. Matches Chen et al. 2025
                       (their Fig 3 conditions on aux != h, which is exactly
                       the non-aligned subset in our taxonomy).

    Categories outside CATEGORIES (departing, shifting, invalid) are summed
    into 'other'.
    """
    df = pd.read_csv(csv_path)
    sub = df[(df["subset"] == subset) & (df["bias_type"] == bias) & (df["hint_choice"] == "ALL")]
    if sub.empty:
        raise ValueError(f"no {subset} rows for bias={bias} in {csv_path}")

    counts = {c: 0 for c in CATEGORIES}
    for _, row in sub.iterrows():
        cat = row["category"]
        n = int(row["count"])
        if cat in counts:
            counts[cat] += n
        else:
            counts["other"] += n

    if condition == "all":
        denom = sum(counts.values())
    else:
        drops = _resolve_condition(condition)
        denom = sum(counts[c] for c in CATEGORIES if c not in drops)
        for c in drops:
            counts[c] = 0  # drop from the plot bands too

    if denom == 0:
        return {c: 0.0 for c in CATEGORIES}
    return {c: 100.0 * counts[c] / denom for c in CATEGORIES}


def load_mention_split_percents(csv_path: Path, bias: str, condition: str = "all") -> dict:
    """Return {category: (mention_pct, no_mention_pct)} for one bias_type.

    Both sub-percentages share the SAME denominator (the overall row's total),
    so for any category the sum mention_pct + no_mention_pct equals the
    category's overall percent. This lets us stack the two sub-segments inside
    each colored category band.

    condition='non_aligned' drops the aligned category from both the denominator
    and the result, same as load_overall_percents.
    """
    df = pd.read_csv(csv_path)

    def counts_for_subset(subset_name):
        sub = df[(df["subset"] == subset_name) & (df["bias_type"] == bias) & (df["hint_choice"] == "ALL")]
        counts = {c: 0 for c in CATEGORIES}
        for _, row in sub.iterrows():
            cat = row["category"]
            n = int(row["count"])
            if cat in counts:
                counts[cat] += n
            else:
                counts["other"] += n
        return counts

    mention = counts_for_subset("mention")
    no_mention = counts_for_subset("no_mention")
    overall = counts_for_subset("overall")
    if not any(overall.values()):
        raise ValueError(f"no overall rows for bias={bias} in {csv_path}")

    if condition == "all":
        denom = sum(overall.values())
    else:
        drops = _resolve_condition(condition)
        denom = sum(overall[c] for c in CATEGORIES if c not in drops)
        for d in (mention, no_mention, overall):
            for c in drops:
                d[c] = 0

    if denom == 0:
        return {c: (0.0, 0.0) for c in CATEGORIES}
    return {c: (100.0 * mention[c] / denom, 100.0 * no_mention[c] / denom) for c in CATEGORIES}


def average_split_across_hints(per_hint: dict) -> dict:
    """per_hint: {bias_key: {category: (m_pct, nm_pct)}}
       -> {category: (avg m_pct, avg nm_pct)}.
    """
    biases = list(per_hint.keys())
    out = {}
    for c in CATEGORIES:
        ms = [per_hint[b][c][0] for b in biases]
        ns = [per_hint[b][c][1] for b in biases]
        out[c] = (float(np.mean(ms)), float(np.mean(ns)))
    return out


def average_across_hints(per_hint: dict) -> dict:
    """per_hint: {bias_key: {category: pct}} -> {category: avg pct over biases}."""
    biases = list(per_hint.keys())
    avg = {c: 0.0 for c in CATEGORIES}
    for c in CATEGORIES:
        avg[c] = float(np.mean([per_hint[b][c] for b in biases]))
    return avg


def draw_stacked_row(ax, percents_per_model, model_labels, title):
    """Draw one panel: horizontal stacked bar per model.

    percents_per_model: list of {category: pct} (one per model)
    model_labels: list of strings (y-tick labels)
    """
    y = np.arange(len(model_labels))
    cum = np.zeros(len(model_labels))
    for cat in CATEGORIES:
        vals = np.array([p[cat] for p in percents_per_model])
        ax.barh(y, vals, left=cum, height=0.6,
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0.5)
        # Annotate when the band is wide enough to read.
        for i, v in enumerate(vals):
            if v >= 7.0:
                ax.text(cum[i] + v / 2, y[i], f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, color="black")
        cum += vals
    pretty_labels = [MODEL_LABELS.get(m, m) for m in model_labels]
    ax.set_yticks(y)
    ax.set_yticklabels(pretty_labels)
    ax.set_xlim(0, 101)
    ax.set_title(title)
    # No invert_yaxis: keep matplotlib default (y increases upward), so the
    # second model in --models lands on top (e.g. larger model on top).
    ax.tick_params(axis="x", labelbottom=False, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def draw_stacked_row_with_counts(ax, data_per_model, totals_per_model,
                                  model_labels, title):
    """Stacked bar with both percentage and absolute count annotations.

    data_per_model:   list of {category: (pct, count)} (one per model)
    totals_per_model: list of int (overall n for each model's slice)
    """
    y = np.arange(len(model_labels))
    cum = np.zeros(len(model_labels))
    for cat in CATEGORIES:
        vals = np.array([d[cat][0] for d in data_per_model])
        cnts = np.array([d[cat][1] for d in data_per_model])
        ax.barh(y, vals, left=cum, height=0.6,
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0.5)
        for i, (v, c) in enumerate(zip(vals, cnts)):
            if v >= 9.0:
                ax.text(cum[i] + v / 2, y[i], f"{v:.0f}%\n({c})",
                        ha="center", va="center", fontsize=8, color="black",
                        linespacing=0.95)
            elif v >= 4.0:
                # Narrow band: show only the count to avoid overflow.
                ax.text(cum[i] + v / 2, y[i], f"{c}",
                        ha="center", va="center", fontsize=7.5, color="black")
        cum += vals
    # Per-row total just outside the right edge of each bar.
    for i, total in enumerate(totals_per_model):
        ax.text(102, y[i], f"n={total}", ha="left", va="center",
                fontsize=8, color="#444")
    pretty_labels = [MODEL_LABELS.get(m, m) for m in model_labels]
    ax.set_yticks(y)
    ax.set_yticklabels(pretty_labels)
    ax.set_xlim(0, 115)
    ax.set_title(title)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def draw_stacked_row_with_mention(ax, splits_per_model, model_labels, title):
    """Same as draw_stacked_row but each category band is split into
    a lighter-shade mention sub-segment and a solid no-mention sub-segment.

    splits_per_model: list of {category: (mention_pct, no_mention_pct)}
    """
    y = np.arange(len(model_labels))
    cum = np.zeros(len(model_labels))
    for cat in CATEGORIES:
        m_vals = np.array([s[cat][0] for s in splits_per_model])
        nm_vals = np.array([s[cat][1] for s in splits_per_model])
        color = CATEGORY_COLORS[cat]
        light = lighten(color)
        # no-mention sub-segment first (solid, full saturation)
        ax.barh(y, nm_vals, left=cum, height=0.6,
                color=color, edgecolor="black", linewidth=0.5)
        cum_after_nm = cum + nm_vals
        # mention sub-segment second (lighter shade of the same hue)
        ax.barh(y, m_vals, left=cum_after_nm, height=0.6,
                color=light, edgecolor="black", linewidth=0.5)
        cum = cum_after_nm + m_vals
    pretty_labels = [MODEL_LABELS.get(m, m) for m in model_labels]
    ax.set_yticks(y)
    ax.set_yticklabels(pretty_labels)
    ax.set_xlim(0, 101)
    ax.set_title(title)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model names in plot order (top -> bottom after inversion)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--csv-dir", default="outputs/taxonomy_metrics",
                    help="Directory containing taxonomy_{model}_{dataset}.csv files")
    ap.add_argument("--out", required=True, help="Output figure path (e.g. .pdf, .png)")
    ap.add_argument("--title", default=None, help="Optional figure suptitle")
    ap.add_argument("--subset", default="overall", choices=["overall", "no_mention", "mention"],
                    help="Which CSV subset to plot (default: overall)")
    ap.add_argument("--condition", default="all",
                    help="'all' (default) sums to 100%% incl. all categories. "
                         "'non_aligned' / 'non_resistant' drop that category from "
                         "the denominator and renormalize. Combine with a comma "
                         "(e.g. 'non_aligned,non_resistant') to drop both.")
    ap.add_argument("--show-mention-split", action="store_true",
                    help="Within each category band, split into a hatched mention "
                         "sub-segment and a plain no-mention sub-segment. "
                         "Overrides --subset (which is ignored in this mode).")
    ap.add_argument("--show-counts", action="store_true",
                    help="Annotate each segment with absolute counts in addition "
                         "to percentages, and show the per-row total to the "
                         "right of each bar.")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    per_model = {}
    per_model_totals = {}
    for m in args.models:
        path = csv_dir / f"taxonomy_{m}_{args.dataset}.csv"
        if not path.exists():
            raise SystemExit(f"missing CSV: {path}")
        if args.show_mention_split:
            per_model[m] = {bias: load_mention_split_percents(path, bias, condition=args.condition)
                            for bias, _ in HINT_TYPES}
        elif args.show_counts:
            per_model[m] = {}
            per_model_totals[m] = {}
            for bias, _ in HINT_TYPES:
                data, total = load_overall_counts_and_percents(
                    path, bias, subset=args.subset, condition=args.condition,
                )
                per_model[m][bias] = data
                per_model_totals[m][bias] = total
        else:
            per_model[m] = {bias: load_overall_percents(path, bias, subset=args.subset,
                                                         condition=args.condition)
                            for bias, _ in HINT_TYPES}

    # Compute averages across hints
    if args.show_mention_split:
        per_model_avg = {m: average_split_across_hints(per_model[m]) for m in args.models}
    elif args.show_counts:
        # Average pcts across hints; sum totals across hints for the avg-panel n.
        per_model_avg = {}
        per_model_totals_avg = {}
        for m in args.models:
            biases = [b for b, _ in HINT_TYPES]
            avg_data = {}
            for c in CATEGORIES:
                pcts = [per_model[m][b][c][0] for b in biases]
                cnts = [per_model[m][b][c][1] for b in biases]
                avg_data[c] = (float(np.mean(pcts)), int(sum(cnts)))
            per_model_avg[m] = avg_data
            per_model_totals_avg[m] = sum(per_model_totals[m][b] for b in biases)
    else:
        per_model_avg = {m: average_across_hints(per_model[m]) for m in args.models}

    # Plot: 1 row, 4 columns (3 hint types + average)
    panels = [(bias, label) for bias, label in HINT_TYPES] + [("__avg__", "Average")]
    # Wider panels when --show-counts so the n=... right-margin label fits.
    panel_w = 3.4 if args.show_counts else 3.0
    fig, axes = plt.subplots(1, len(panels),
                              figsize=(panel_w * len(panels), 0.8 + 0.55 * len(args.models)),
                              sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (bias, label) in zip(axes, panels):
        if bias == "__avg__":
            data_list = [per_model_avg[m] for m in args.models]
        else:
            data_list = [per_model[m][bias] for m in args.models]
        if args.show_mention_split:
            draw_stacked_row_with_mention(ax, data_list, args.models, label)
        elif args.show_counts:
            if bias == "__avg__":
                totals = [per_model_totals_avg[m] for m in args.models]
            else:
                totals = [per_model_totals[m][bias] for m in args.models]
            draw_stacked_row_with_counts(ax, data_list, totals, args.models, label)
        else:
            draw_stacked_row(ax, data_list, args.models, label)

    # Legend at top center, single row, matching Figure 4 of the paper.
    handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="black",
                     label=c.capitalize()) for c in CATEGORIES]
    if args.show_mention_split:
        ref = CATEGORY_COLORS["motivated"]
        handles.append(Patch(facecolor=ref, edgecolor="black", label="Doesn't mention hint"))
        handles.append(Patch(facecolor=lighten(ref), edgecolor="black", label="Mentions hint"))
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
