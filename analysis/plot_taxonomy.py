#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CATEGORIES = ["motivated", "resistant", "aligned", "departing", "shifting", "invalid"]
CATEGORY_COLORS = {
    "motivated": "#FF6B6B",  # Modern coral red
    "resistant": "#51CF66",  # Fresh green
    "aligned": "#4DABF7",  # Modern teal blue
    "departing": "#FFD43B",  # Warm gold (swapped)
    "shifting": "#B197FC",  # Soft purple (swapped)
    "invalid": "#adb5bd",  # neutral grey
}
STYLE_CHOICES = ["hatched"]

# Label mappings
BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}

MODEL_LABELS = {
    "gemma-3-4b": "Gemma-3-4B-it",
    "qwen-3-8b": "Qwen-3-8B-thinking",
    "llama-3.1-8b": "Llama-3.1-8B-Instruct",
}

DATASET_LABELS = {
    "aqua": "AQUA",
    "mmlu": "MMLU",
    "arc-challenge": "ARC-Challenge",
    "commonsense_qa": "CommonsenseQA",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot hint taxonomy breakdown for a model/dataset pair.")
    parser.add_argument("--model", help="Model name, e.g. qwen-3-8b")
    parser.add_argument("--dataset", help="Dataset name, e.g. arc-challenge")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all available model-dataset combinations",
    )
    parser.add_argument(
        "--metrics-dir",
        default=os.path.join("outputs", "taxonomy_metrics"),
        help="Directory containing taxonomy_*.csv files",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join("figures", "taxonomy"),
        help="Directory to store generated figures",
    )
    parser.add_argument(
        "--fmt",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format/extension for saved plots",
    )
    parser.add_argument(
        "--plot-per-hint",
        action="store_true",
        help="Also create a per-hint breakdown figure (one panel per bias type).",
    )
    parser.add_argument(
        "--style",
        action="append",
        choices=STYLE_CHOICES,
        help="Visualization style for mention/no-mention split. Provide multiple times for multiple styles.",
    )
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Generate figures for every available style.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Generate an aggregate plot averaging across all model-dataset combinations with percentages.",
    )
    parser.add_argument(
        "--aggregate-by-dataset",
        action="store_true",
        help="Generate aggregate plot with one bar per dataset (averaging over models and hint types).",
    )
    parser.add_argument(
        "--aggregate-by-model",
        action="store_true",
        help="Generate aggregate plot with one bar per model (averaging over datasets and hint types).",
    )
    return parser.parse_args()


def ensure_category_order(df):
    for cat in CATEGORIES:
        if cat not in df:
            df[cat] = 0
    return df[CATEGORIES]


def load_taxonomy(path):
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {path}")
    df = pd.read_csv(path)
    required_cols = {"bias_type", "subset", "category", "count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Taxonomy CSV missing columns: {missing}")
    return df


def stacked_bar(ax, pivot_df, title):
    if pivot_df.empty:
        ax.set_visible(False)
        return
    x = np.arange(len(pivot_df.index))
    width = 0.7
    cumulative = np.zeros(len(pivot_df.index))
    for cat in CATEGORIES:
        values = pivot_df[cat].values
        ax.bar(x, values, width=width, bottom=cumulative, label=cat)
        cumulative += values
    ax.set_xticks(x)
    ax.set_xticklabels([label.title() for label in pivot_df.index], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title(title)


def average_percentages(subset_df, overall_totals, group_label="bias_type"):
    """
    Compute per-group (bias or dataset) articulation percentages by:
    1) converting counts for each model/dataset/bias into percentages using the overall total
    2) averaging those percentages across model/dataset combinations
    """
    subset_groups = {
        key: group for key, group in subset_df.groupby(["model", "dataset", "bias_type"])
    }
    rows = []
    for key, total in overall_totals.items():
        model, dataset, bias = key
        group_map = {
            "bias_type": bias,
            "dataset": dataset,
            "model": model,
        }
        group_value = group_map.get(group_label, bias)
        if total <= 0:
            continue
        group = subset_groups.get(key)
        row = {group_label: group_value}
        for cat in CATEGORIES:
            count = 0.0
            if group is not None:
                count = group[group["category"] == cat]["count"].sum()
            row[cat] = (count / total) * 100
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[group_label] + CATEGORIES)

    pct_df = pd.DataFrame(rows)
    return pct_df.groupby(group_label)[CATEGORIES].mean().reset_index()


def get_subset_counts(df, bias, subset):
    sub = df[(df["subset"] == subset) & (df["bias_type"] == bias)]
    if sub.empty:
        return pd.Series(0, index=CATEGORIES)
    return sub.set_index("category")["count"].reindex(CATEGORIES, fill_value=0)


def compute_style_data(df, biases):
    data = {}
    for cat in CATEGORIES:
        mention_vals = []
        no_vals = []
        for bias in biases:
            mention_vals.append(get_subset_counts(df, bias, "mention")[cat])
            no_vals.append(get_subset_counts(df, bias, "no_mention")[cat])
        data[cat] = (np.array(mention_vals), np.array(no_vals))
    return data


def draw_style_hatched(ax, x, width, biases, data):
    cumulative = np.zeros(len(biases))
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        mention_vals, no_vals = data[cat]
        ax.bar(x, mention_vals, width=width, bottom=cumulative, color=color, edgecolor="black", hatch="//")
        cumulative += mention_vals
        ax.bar(x, no_vals, width=width, bottom=cumulative, color=color, edgecolor="black")
        cumulative += no_vals
    return cumulative.max()


def draw_style_paired(ax, x, width, biases, data):
    left = x - width / 3
    right = x + width / 3
    bar_width = width / 2.2
    cum_left = np.zeros(len(biases))
    cum_right = np.zeros(len(biases))
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        mention_vals, no_vals = data[cat]
        ax.bar(left, mention_vals, width=bar_width, bottom=cum_left, color=color, edgecolor="black")
        cum_left += mention_vals
        ax.bar(right, no_vals, width=bar_width, bottom=cum_right, color=color, edgecolor="black")
        cum_right += no_vals
    return max(cum_left.max(), cum_right.max())


def draw_style_alpha(ax, x, width, biases, data):
    cumulative = np.zeros(len(biases))
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        mention_vals, no_vals = data[cat]
        ax.bar(x, mention_vals, width=width, bottom=cumulative, color=color, edgecolor="black", alpha=0.95)
        cumulative += mention_vals
        ax.bar(x, no_vals, width=width, bottom=cumulative, color=color, edgecolor="black", alpha=0.35)
        cumulative += no_vals
    return cumulative.max()


def draw_style_border(ax, x, width, biases, data):
    cumulative = np.zeros(len(biases))
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        mention_vals, no_vals = data[cat]
        ax.bar(
            x,
            mention_vals,
            width=width,
            bottom=cumulative,
            color=color,
            edgecolor="black",
            linewidth=1.6,
        )
        cumulative += mention_vals
        ax.bar(
            x,
            no_vals,
            width=width,
            bottom=cumulative,
            color=color,
            edgecolor="none",
            linewidth=0,
        )
        cumulative += no_vals
    return cumulative.max()


def add_dot_overlay(ax, rect, spacing_x=0.08, spacing_y=0.2):
    height = rect.get_height()
    if height <= 0:
        return
    x0 = rect.get_x()
    width = rect.get_width()
    y0 = rect.get_y()
    xs = np.arange(x0 + spacing_x / 2, x0 + width, spacing_x)
    ys = np.arange(y0 + spacing_y / 2, y0 + height, spacing_y)
    if len(xs) == 0 or len(ys) == 0:
        return
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(X.ravel(), Y.ravel(), s=8, color="white", alpha=0.85, linewidths=0)


def draw_style_dots(ax, x, width, biases, data):
    cumulative = np.zeros(len(biases))
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        mention_vals, no_vals = data[cat]
        mention_rects = ax.bar(
            x,
            mention_vals,
            width=width,
            bottom=cumulative,
            color=color,
            edgecolor="black",
        )
        cumulative += mention_vals
        for rect in mention_rects:
            add_dot_overlay(ax, rect)
        ax.bar(x, no_vals, width=width, bottom=cumulative, color=color, edgecolor="black")
        cumulative += no_vals
    return cumulative.max()


STYLE_DRAWERS = {
    "hatched": draw_style_hatched,
    "paired": draw_style_paired,
    "alpha": draw_style_alpha,
    "border": draw_style_border,
    "dots": draw_style_dots,
}


def pattern_handles_for_style(style):
    if style == "hatched":
        return [
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Mention"),
            Patch(facecolor="white", edgecolor="black", label="No mention"),
        ]
    if style == "paired":
        return [
            Patch(facecolor="white", edgecolor="black", label="Mention (left stack)"),
            Patch(facecolor="white", edgecolor="black", label="No mention (right stack)"),
        ]
    if style == "alpha":
        return [
            Patch(facecolor="gray", edgecolor="black", alpha=0.95, label="Mention"),
            Patch(facecolor="gray", edgecolor="black", alpha=0.35, label="No mention"),
        ]
    if style == "border":
        return [
            Patch(facecolor="white", edgecolor="black", linewidth=1.5, label="Mention (outlined)"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.2, label="No mention"),
        ]
    if style == "dots":
        return [
            Line2D([0], [0], marker="o", color="white", markerfacecolor="black", label="Mention (dot overlay)", markersize=6),
            Patch(facecolor="white", edgecolor="black", label="No mention"),
        ]
    return [
        Patch(facecolor="white", edgecolor="black", label="Mention"),
        Patch(facecolor="white", edgecolor="black", label="No mention"),
    ]


def plot_overall_style(df, model, dataset, save_path, style):
    biases = sorted(df["bias_type"].unique())
    if not biases:
        print("No bias rows found; skipping plot.")
        return
    data = compute_style_data(df, biases)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.6
    x = np.arange(len(biases))
    drawer = STYLE_DRAWERS[style]
    max_height = drawer(ax, x, width, biases, data)

    ax.set_xticks(x)
    bias_labels = [BIAS_LABELS.get(b, b.title()) for b in biases]
    ax.set_xticklabels(bias_labels, rotation=15, ha="center")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max_height * 1.05 if max_height > 0 else 1)
    model_label = MODEL_LABELS.get(model, model)
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    ax.set_title(f"{model_label} on {dataset_label}")

    category_handles = [
        Patch(facecolor=CATEGORY_COLORS.get(cat, "#ccc"), label=cat.title(), edgecolor="black")
        for cat in CATEGORIES
    ]
    legend1 = ax.legend(handles=category_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Category")
    ax.add_artist(legend1)
    ax.legend(
        handles=pattern_handles_for_style(style),
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
        title="Articulation",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved taxonomy plot ({style}) to {save_path}")
    plt.close(fig)


def plot_per_hint(df, model, dataset, save_path):
    per_hint = df[df["subset"] == "per_hint"]
    if per_hint.empty:
        print("No per-hint entries found; skipping per-hint plot.")
        return
    biases = sorted(per_hint["bias_type"].unique())
    n_rows = len(biases)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 4 * n_rows), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, bias in zip(axes, biases):
        sub = per_hint[per_hint["bias_type"] == bias]
        pivot = (
            sub.pivot(index="hint_choice", columns="category", values="count")
            .sort_index()
            .fillna(0)
        )
        pivot = ensure_category_order(pivot)
        bias_label = BIAS_LABELS.get(bias, bias.title())
        stacked_bar(ax, pivot, f"{bias_label} hints")
        ax.set_xlabel("Hint Choice")

    model_label = MODEL_LABELS.get(model, model)
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    fig.suptitle(f"Per-Hint — {model_label} on {dataset_label}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved per-hint taxonomy plot to {save_path}")
    plt.close(fig)


def plot_aggregate(metrics_dir, save_path, style="hatched"):
    """Generate aggregate plot averaging across all model-dataset combinations."""
    metrics_dir = Path(metrics_dir)
    csv_files = list(metrics_dir.glob("taxonomy_*.csv"))
    if not csv_files:
        print(f"No taxonomy CSV files found in {metrics_dir}")
        return
    
    # Load all data
    all_data = []
    for csv_file in csv_files:
        df = load_taxonomy(csv_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Compute per-bias averages (percentages) for each subset
    overall_data = combined_df[combined_df["subset"] == "overall"].copy()
    mention_data = combined_df[combined_df["subset"] == "mention"].copy()
    no_mention_data = combined_df[combined_df["subset"] == "no_mention"].copy()

    # Total counts per model/dataset/bias for normalization
    overall_totals = {
        key: group["count"].sum()
        for key, group in overall_data.groupby(["model", "dataset", "bias_type"])
    }

    mention_avg = average_percentages(mention_data, overall_totals, group_label="bias_type")
    no_mention_avg = average_percentages(no_mention_data, overall_totals, group_label="bias_type")

    biases = sorted(overall_data["bias_type"].unique())
    data = {}
    for cat in CATEGORIES:
        mention_vals = []
        no_vals = []
        for bias in biases:
            mention_pct = mention_avg[(mention_avg["bias_type"] == bias)][cat].values
            no_pct = no_mention_avg[(no_mention_avg["bias_type"] == bias)][cat].values
            mention_vals.append(mention_pct[0] if len(mention_pct) > 0 else 0.0)
            no_vals.append(no_pct[0] if len(no_pct) > 0 else 0.0)
        data[cat] = (np.array(mention_vals), np.array(no_vals))
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.6
    x = np.arange(len(biases))
    drawer = STYLE_DRAWERS[style]
    max_height = drawer(ax, x, width, biases, data)  # Draw bars with percentage values
    
    ax.set_xticks(x)
    bias_labels = [BIAS_LABELS.get(b, b.title()) for b in biases]
    ax.set_xticklabels(bias_labels, rotation=15, ha="center")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Taxonomy per Hint Type")
    
    category_handles = [
        Patch(facecolor=CATEGORY_COLORS.get(cat, "#ccc"), label=cat.title(), edgecolor="black")
        for cat in CATEGORIES
    ]
    legend1 = ax.legend(handles=category_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Category")
    ax.add_artist(legend1)
    ax.legend(
        handles=pattern_handles_for_style(style),
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
        title="Articulation",
    )
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved aggregate taxonomy plot to {save_path}")
    plt.close(fig)


def plot_dataset_aggregate(metrics_dir, save_path, style="hatched"):
    """Generate aggregate plot with one bar per dataset (averaging across models and hint types)."""
    metrics_dir = Path(metrics_dir)
    csv_files = list(metrics_dir.glob("taxonomy_*.csv"))
    if not csv_files:
        print(f"No taxonomy CSV files found in {metrics_dir}")
        return

    all_data = [load_taxonomy(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(all_data, ignore_index=True)

    overall_data = combined_df[combined_df["subset"] == "overall"].copy()
    mention_data = combined_df[combined_df["subset"] == "mention"].copy()
    no_mention_data = combined_df[combined_df["subset"] == "no_mention"].copy()

    overall_totals = {
        key: group["count"].sum()
        for key, group in overall_data.groupby(["model", "dataset", "bias_type"])
    }

    mention_avg = average_percentages(mention_data, overall_totals, group_label="dataset")
    no_mention_avg = average_percentages(no_mention_data, overall_totals, group_label="dataset")

    datasets = sorted(mention_avg["dataset"].unique())
    data = {}
    for cat in CATEGORIES:
        mention_vals = []
        no_vals = []
        for dataset in datasets:
            mention_pct = mention_avg[mention_avg["dataset"] == dataset][cat].values
            no_pct = no_mention_avg[no_mention_avg["dataset"] == dataset][cat].values
            mention_vals.append(mention_pct[0] if len(mention_pct) > 0 else 0.0)
            no_vals.append(no_pct[0] if len(no_pct) > 0 else 0.0)
        data[cat] = (np.array(mention_vals), np.array(no_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.6
    x = np.arange(len(datasets))
    drawer = STYLE_DRAWERS[style]
    drawer(ax, x, width, datasets, data)

    ax.set_xticks(x)
    dataset_labels = [DATASET_LABELS.get(ds, ds.replace("_", " ").title()) for ds in datasets]
    ax.set_xticklabels(dataset_labels, rotation=15, ha="center")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Taxonomy per Dataset")

    category_handles = [
        Patch(facecolor=CATEGORY_COLORS.get(cat, "#ccc"), label=cat.title(), edgecolor="black")
        for cat in CATEGORIES
    ]
    legend1 = ax.legend(handles=category_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Category")
    ax.add_artist(legend1)
    ax.legend(
        handles=pattern_handles_for_style(style),
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
        title="Articulation",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved dataset aggregate taxonomy plot to {save_path}")
    plt.close(fig)


def plot_model_aggregate(metrics_dir, save_path, style="hatched"):
    """Generate aggregate plot with one bar per model (averaging across datasets and hint types)."""
    metrics_dir = Path(metrics_dir)
    csv_files = list(metrics_dir.glob("taxonomy_*.csv"))
    if not csv_files:
        print(f"No taxonomy CSV files found in {metrics_dir}")
        return

    all_data = [load_taxonomy(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(all_data, ignore_index=True)

    overall_data = combined_df[combined_df["subset"] == "overall"].copy()
    mention_data = combined_df[combined_df["subset"] == "mention"].copy()
    no_mention_data = combined_df[combined_df["subset"] == "no_mention"].copy()

    overall_totals = {
        key: group["count"].sum()
        for key, group in overall_data.groupby(["model", "dataset", "bias_type"])
    }

    mention_avg = average_percentages(mention_data, overall_totals, group_label="model")
    no_mention_avg = average_percentages(no_mention_data, overall_totals, group_label="model")

    models = sorted(mention_avg["model"].unique())
    data = {}
    for cat in CATEGORIES:
        mention_vals = []
        no_vals = []
        for model in models:
            mention_pct = mention_avg[mention_avg["model"] == model][cat].values
            no_pct = no_mention_avg[no_mention_avg["model"] == model][cat].values
            mention_vals.append(mention_pct[0] if len(mention_pct) > 0 else 0.0)
            no_vals.append(no_pct[0] if len(no_pct) > 0 else 0.0)
        data[cat] = (np.array(mention_vals), np.array(no_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.6
    x = np.arange(len(models))
    drawer = STYLE_DRAWERS[style]
    drawer(ax, x, width, models, data)

    ax.set_xticks(x)
    model_labels = [MODEL_LABELS.get(m, m.replace("_", " ").title()) for m in models]
    ax.set_xticklabels(model_labels, rotation=15, ha="center")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Taxonomy per Model")

    category_handles = [
        Patch(facecolor=CATEGORY_COLORS.get(cat, "#ccc"), label=cat.title(), edgecolor="black")
        for cat in CATEGORIES
    ]
    legend1 = ax.legend(handles=category_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Category")
    ax.add_artist(legend1)
    ax.legend(
        handles=pattern_handles_for_style(style),
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
        title="Articulation",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved model aggregate taxonomy plot to {save_path}")
    plt.close(fig)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        # Generate aggregate plot
        default_invocation = not args.all_styles and not args.style
        if args.all_styles:
            styles_to_plot = STYLE_CHOICES
        elif args.style:
            styles_to_plot = args.style
        else:
            styles_to_plot = ["hatched"]
        
        for style in styles_to_plot:
            if default_invocation and style == "hatched":
                filename = f"taxonomy_aggregate.{args.fmt}"
            else:
                filename = f"taxonomy_aggregate_{style}.{args.fmt}"
            plot_aggregate(args.metrics_dir, save_dir / filename, style)
        return

    if args.aggregate_by_dataset:
        default_invocation = not args.all_styles and not args.style
        if args.all_styles:
            styles_to_plot = STYLE_CHOICES
        elif args.style:
            styles_to_plot = args.style
        else:
            styles_to_plot = ["hatched"]

        for style in styles_to_plot:
            if default_invocation and style == "hatched":
                filename = f"taxonomy_dataset_aggregate.{args.fmt}"
            else:
                filename = f"taxonomy_dataset_aggregate_{style}.{args.fmt}"
            plot_dataset_aggregate(args.metrics_dir, save_dir / filename, style)
        return

    if args.aggregate_by_model:
        default_invocation = not args.all_styles and not args.style
        if args.all_styles:
            styles_to_plot = STYLE_CHOICES
        elif args.style:
            styles_to_plot = args.style
        else:
            styles_to_plot = ["hatched"]

        for style in styles_to_plot:
            if default_invocation and style == "hatched":
                filename = f"taxonomy_model_aggregate.{args.fmt}"
            else:
                filename = f"taxonomy_model_aggregate_{style}.{args.fmt}"
            plot_model_aggregate(args.metrics_dir, save_dir / filename, style)
        return

    if args.all:
        # Find all taxonomy CSV files
        metrics_dir = Path(args.metrics_dir)
        csv_files = list(metrics_dir.glob("taxonomy_*.csv"))
        if not csv_files:
            print(f"No taxonomy CSV files found in {metrics_dir}")
            return
        
        # Extract model and dataset from filenames
        pattern = re.compile(r"taxonomy_(.+?)_(.+?)\.csv$")
        combinations = []
        for csv_file in csv_files:
            match = pattern.match(csv_file.name)
            if match:
                model, dataset = match.groups()
                combinations.append((model, dataset, csv_file))
        
        print(f"Found {len(combinations)} model-dataset combinations")
        for model, dataset, csv_file in combinations:
            print(f"Processing {model} on {dataset}...")
            df = load_taxonomy(csv_file)
            
            default_invocation = not args.all_styles and not args.style
            if args.all_styles:
                styles_to_plot = STYLE_CHOICES
            elif args.style:
                styles_to_plot = args.style
            else:
                styles_to_plot = ["hatched"]
            
            for style in styles_to_plot:
                if default_invocation and style == "hatched":
                    filename = f"taxonomy_{model}_{dataset}.{args.fmt}"
                else:
                    filename = f"taxonomy_{model}_{dataset}_{style}.{args.fmt}"
                plot_overall_style(df, model, dataset, save_dir / filename, style)
            
            if args.plot_per_hint:
                per_hint_filename = f"taxonomy_{model}_{dataset}_per_hint.{args.fmt}"
                plot_per_hint(df, model, dataset, save_dir / per_hint_filename)
    else:
        # Single model-dataset processing
        if not args.model or not args.dataset:
            import sys
            print("Error: --model and --dataset are required unless --all is specified", file=sys.stderr)
            sys.exit(1)
        
        metrics_path = Path(args.metrics_dir) / f"taxonomy_{args.model}_{args.dataset}.csv"
        df = load_taxonomy(metrics_path)

        default_invocation = not args.all_styles and not args.style
        if args.all_styles:
            styles_to_plot = STYLE_CHOICES
        elif args.style:
            styles_to_plot = args.style
        else:
            styles_to_plot = ["hatched"]

        for style in styles_to_plot:
            if default_invocation and style == "hatched":
                filename = f"taxonomy_{args.model}_{args.dataset}.{args.fmt}"
            else:
                filename = f"taxonomy_{args.model}_{args.dataset}_{style}.{args.fmt}"
            plot_overall_style(df, args.model, args.dataset, save_dir / filename, style)

        if args.plot_per_hint:
            per_hint_filename = f"taxonomy_{args.model}_{args.dataset}_per_hint.{args.fmt}"
            plot_per_hint(df, args.model, args.dataset, save_dir / per_hint_filename)


if __name__ == "__main__":
    main()

