#!/usr/bin/env python3
"""
Plot cross-dataset and cross-hint transfer matrices from probe evaluation results.

Reads the probe_metrics SQLite DB and produces:
1. Cross-dataset transfer heatmap (4x4 matrix, averaged across models and biases)
2. Cross-hint transfer heatmap (3x3 matrix, averaged across models and datasets)
3. Per-model cross-dataset heatmaps
4. Permutation baseline comparison bar chart

Usage:
    python analysis/plot_transfer_matrix.py
    python analysis/plot_transfer_matrix.py --db /path/to/probe_metrics.db
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

MOTIVATION_HOME = Path(os.environ.get("MOTIVATION_HOME", "outputs"))
DEFAULT_DB = MOTIVATION_HOME / "probe_metrics.db"
FIGURES_DIR = MOTIVATION_HOME / "figures"

DATASET_LABELS = {
    "mmlu": "MMLU",
    "arc-challenge": "ARC",
    "commonsense_qa": "CSQA",
    "aqua": "AQuA",
}
DATASET_ORDER = ["mmlu", "arc-challenge", "commonsense_qa", "aqua"]

BIAS_LABELS = {
    "expert": "Sycophancy",
    "self": "Consistency",
    "metadata": "Metadata",
}
BIAS_ORDER = ["expert", "self", "metadata"]

MODEL_LABELS = {
    "qwen-3-8b": "Qwen-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gemma-3-4b": "Gemma-3-4B",
}
MODEL_ORDER = ["qwen-3-8b", "llama-3.1-8b", "gemma-3-4b"]


def load_results(db_path):
    """Load all probe metrics into a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM probe_metrics", conn)
    conn.close()
    return df


def parse_cross_tag(tag):
    """Parse tag like 'eval_dataset=arc-challenge_eval_bias=self' into a dict."""
    parts = {}
    if not tag:
        return parts
    for match in re.finditer(r'(eval_\w+)=([^_]+(?:_[^_=]+)*?)(?=_eval_|_permuted|$)', tag):
        key, value = match.group(1), match.group(2)
        # Handle dataset names with hyphens (arc-challenge) and underscores (commonsense_qa)
        parts[key] = value
    if 'permuted_eval' in tag:
        parts['permuted'] = True
    return parts


def get_best_auc(df, groupby_cols, classifier='rfm', probe='mot_vs_alg'):
    """Get best AUC (across layers/steps) for each group."""
    mask = (df['classifier'] == classifier) & (df['probe'] == probe)
    filtered = df[mask]
    return filtered.groupby(groupby_cols)['auc'].max().reset_index()


def build_cross_dataset_matrix(df, classifier='rfm', probe='mot_vs_alg'):
    """Build a (train_dataset x eval_dataset) AUC matrix, averaged across models and biases."""
    # Standard results (tag is empty): train_dataset == eval_dataset (diagonal)
    standard = df[df['tag'] == ''].copy()
    standard_best = get_best_auc(standard, ['model', 'dataset', 'bias'], classifier, probe)

    # Cross-dataset results: tag contains eval_dataset=X
    cross = df[df['tag'].str.contains('eval_dataset=', na=False) &
               ~df['tag'].str.contains('permuted', na=False)].copy()
    cross['eval_dataset'] = cross['tag'].apply(
        lambda t: parse_cross_tag(t).get('eval_dataset', ''))
    cross = cross[cross['eval_dataset'] != '']
    cross_best = get_best_auc(cross, ['model', 'dataset', 'bias', 'eval_dataset'], classifier, probe)
    cross_best = cross_best.rename(columns={'dataset': 'train_dataset'})

    # Build matrix
    matrix = np.full((len(DATASET_ORDER), len(DATASET_ORDER)), np.nan)
    for i, train_ds in enumerate(DATASET_ORDER):
        # Diagonal: standard results
        diag_rows = standard_best[standard_best['dataset'] == train_ds]
        if len(diag_rows) > 0:
            matrix[i, i] = diag_rows['auc'].mean()
        # Off-diagonal: cross results
        for j, eval_ds in enumerate(DATASET_ORDER):
            if i == j:
                continue
            cross_rows = cross_best[
                (cross_best['train_dataset'] == train_ds) &
                (cross_best['eval_dataset'] == eval_ds)
            ]
            if len(cross_rows) > 0:
                matrix[i, j] = cross_rows['auc'].mean()

    return matrix


def build_cross_hint_matrix(df, classifier='rfm', probe='mot_vs_alg'):
    """Build a (train_bias x eval_bias) AUC matrix, averaged across models and datasets."""
    # Standard results (diagonal)
    standard = df[df['tag'] == ''].copy()
    standard_best = get_best_auc(standard, ['model', 'dataset', 'bias'], classifier, probe)

    # Cross-hint results
    cross = df[df['tag'].str.contains('eval_bias=', na=False) &
               ~df['tag'].str.contains('permuted', na=False)].copy()
    cross['eval_bias'] = cross['tag'].apply(
        lambda t: parse_cross_tag(t).get('eval_bias', ''))
    cross = cross[cross['eval_bias'] != '']
    cross_best = get_best_auc(cross, ['model', 'dataset', 'bias', 'eval_bias'], classifier, probe)
    cross_best = cross_best.rename(columns={'bias': 'train_bias'})

    matrix = np.full((len(BIAS_ORDER), len(BIAS_ORDER)), np.nan)
    for i, train_b in enumerate(BIAS_ORDER):
        # Diagonal
        diag_rows = standard_best[standard_best['bias'] == train_b]
        if len(diag_rows) > 0:
            matrix[i, i] = diag_rows['auc'].mean()
        # Off-diagonal
        for j, eval_b in enumerate(BIAS_ORDER):
            if i == j:
                continue
            cross_rows = cross_best[
                (cross_best['train_bias'] == train_b) &
                (cross_best['eval_bias'] == eval_b)
            ]
            if len(cross_rows) > 0:
                matrix[i, j] = cross_rows['auc'].mean()

    return matrix


def plot_heatmap(matrix, labels, title, filename, label_map=None):
    """Plot a transfer heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    display_labels = [label_map.get(l, l) if label_map else l for l in labels]

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_yticklabels(display_labels)
    ax.set_xlabel('Eval')
    ax.set_ylabel('Train')
    ax.set_title(title)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.65 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=12)

    plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    print(f"Saved {FIGURES_DIR / filename}")
    plt.close()


def plot_permutation_comparison(df, classifier='rfm', probe='mot_vs_alg'):
    """Bar chart: real vs permuted AUC."""
    # Permuted results
    perm = df[df['tag'].str.contains('permuted_eval', na=False)].copy()
    perm_best = get_best_auc(perm, ['model', 'dataset', 'bias'], classifier, probe)

    # Standard results
    standard = df[df['tag'] == ''].copy()
    std_best = get_best_auc(standard, ['model', 'dataset', 'bias'], classifier, probe)

    if len(perm_best) == 0:
        print("No permutation results found, skipping plot.")
        return

    real_mean = std_best['auc'].mean()
    real_std = std_best['auc'].std()
    perm_mean = perm_best['auc'].mean()
    perm_std = perm_best['auc'].std()

    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(['Real Labels', 'Permuted Labels'], [real_mean, perm_mean],
                  yerr=[real_std, perm_std], capsize=5,
                  color=['#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('AUC')
    ax.set_title('Permutation Baseline')
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()

    for bar, val in zip(bars, [real_mean, perm_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=12)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'permutation_baseline.png', dpi=150, bbox_inches='tight')
    print(f"Saved {FIGURES_DIR / 'permutation_baseline.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default=str(DEFAULT_DB))
    parser.add_argument('--classifier', type=str, default='rfm')
    parser.add_argument('--probe', type=str, default='mot_vs_alg')
    args = parser.parse_args()

    print(f"Loading results from {args.db}")
    df = load_results(args.db)
    print(f"Total rows: {len(df)}")

    tags_with_data = df[df['tag'] != '']['tag'].unique()
    print(f"Distinct non-empty tags: {len(tags_with_data)}")
    for t in sorted(tags_with_data)[:10]:
        print(f"  {t}")

    # Cross-dataset transfer matrix
    print("\n=== Cross-dataset transfer ===")
    xd_matrix = build_cross_dataset_matrix(df, args.classifier, args.probe)
    print(pd.DataFrame(xd_matrix,
        index=[DATASET_LABELS[d] for d in DATASET_ORDER],
        columns=[DATASET_LABELS[d] for d in DATASET_ORDER]).to_string())
    plot_heatmap(xd_matrix, DATASET_ORDER,
                 'Cross-Dataset Transfer (AUC)', 'cross_dataset_transfer.png',
                 DATASET_LABELS)

    # Cross-hint transfer matrix
    print("\n=== Cross-hint transfer ===")
    xb_matrix = build_cross_hint_matrix(df, args.classifier, args.probe)
    print(pd.DataFrame(xb_matrix,
        index=[BIAS_LABELS[b] for b in BIAS_ORDER],
        columns=[BIAS_LABELS[b] for b in BIAS_ORDER]).to_string())
    plot_heatmap(xb_matrix, BIAS_ORDER,
                 'Cross-Hint Transfer (AUC)', 'cross_hint_transfer.png',
                 BIAS_LABELS)

    # Permutation baseline
    print("\n=== Permutation baseline ===")
    plot_permutation_comparison(df, args.classifier, args.probe)


if __name__ == '__main__':
    main()
