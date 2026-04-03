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
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
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


def get_last_layer_auc(df, groupby_cols, classifier='rfm', probe='mot_vs_alg', step=None):
    """Get AUC at the last layer for each group, optionally at a specific step."""
    mask = (df['classifier'] == classifier) & (df['probe'] == probe)
    if step is not None:
        mask = mask & (df['step'] == step)
    filtered = df[mask].copy()
    # Last layer = max layer per model
    last_layers = filtered.groupby('model')['layer'].max().reset_index()
    last_layers.columns = ['model', 'last_layer']
    filtered = filtered.merge(last_layers, on='model')
    filtered = filtered[filtered['layer'] == filtered['last_layer']]
    return filtered.groupby(groupby_cols)['auc'].mean().reset_index()


def build_cross_dataset_matrix(df, classifier='rfm', probe='mot_vs_alg', step=None):
    """Build a (train_dataset x eval_dataset) AUC matrix, averaged across models and biases."""
    # Standard results (tag is empty): train_dataset == eval_dataset (diagonal)
    standard = df[df['tag'] == ''].copy()
    standard_auc = get_last_layer_auc(standard, ['model', 'dataset', 'bias'], classifier, probe, step)

    # Cross-dataset results: tag contains eval_dataset=X
    cross = df[df['tag'].str.contains('eval_dataset=', na=False) &
               ~df['tag'].str.contains('permuted', na=False)].copy()
    cross['eval_dataset'] = cross['tag'].apply(
        lambda t: parse_cross_tag(t).get('eval_dataset', ''))
    cross = cross[cross['eval_dataset'] != '']
    cross_auc = get_last_layer_auc(cross, ['model', 'dataset', 'bias', 'eval_dataset'], classifier, probe, step)
    cross_auc = cross_auc.rename(columns={'dataset': 'train_dataset'})

    # Build matrix
    matrix = np.full((len(DATASET_ORDER), len(DATASET_ORDER)), np.nan)
    for i, train_ds in enumerate(DATASET_ORDER):
        # Diagonal: standard results
        diag_rows = standard_auc[standard_auc['dataset'] == train_ds]
        if len(diag_rows) > 0:
            matrix[i, i] = diag_rows['auc'].mean()
        # Off-diagonal: cross results
        for j, eval_ds in enumerate(DATASET_ORDER):
            if i == j:
                continue
            cross_rows = cross_auc[
                (cross_auc['train_dataset'] == train_ds) &
                (cross_auc['eval_dataset'] == eval_ds)
            ]
            if len(cross_rows) > 0:
                matrix[i, j] = cross_rows['auc'].mean()

    return matrix


def build_cross_hint_matrix(df, classifier='rfm', probe='mot_vs_alg', step=None):
    """Build a (train_bias x eval_bias) AUC matrix, averaged across models and datasets."""
    # Standard results (diagonal)
    standard = df[df['tag'] == ''].copy()
    standard_auc = get_last_layer_auc(standard, ['model', 'dataset', 'bias'], classifier, probe, step)

    # Cross-hint results
    cross = df[df['tag'].str.contains('eval_bias=', na=False) &
               ~df['tag'].str.contains('permuted', na=False)].copy()
    cross['eval_bias'] = cross['tag'].apply(
        lambda t: parse_cross_tag(t).get('eval_bias', ''))
    cross = cross[cross['eval_bias'] != '']
    cross_auc = get_last_layer_auc(cross, ['model', 'dataset', 'bias', 'eval_bias'], classifier, probe, step)
    cross_auc = cross_auc.rename(columns={'bias': 'train_bias'})

    matrix = np.full((len(BIAS_ORDER), len(BIAS_ORDER)), np.nan)
    for i, train_b in enumerate(BIAS_ORDER):
        # Diagonal
        diag_rows = standard_auc[standard_auc['bias'] == train_b]
        if len(diag_rows) > 0:
            matrix[i, i] = diag_rows['auc'].mean()
        # Off-diagonal
        for j, eval_b in enumerate(BIAS_ORDER):
            if i == j:
                continue
            cross_rows = cross_auc[
                (cross_auc['train_bias'] == train_b) &
                (cross_auc['eval_bias'] == eval_b)
            ]
            if len(cross_rows) > 0:
                matrix[i, j] = cross_rows['auc'].mean()

    return matrix


def build_cross_model_matrix(df, classifier='rfm', probe='mot_vs_alg', step=None):
    """Build a (train_model x eval_model) AUC matrix, averaged across datasets and biases.

    Only includes models that have cross-model results (same hidden dimension).
    """
    # Standard results (diagonal)
    standard = df[df['tag'] == ''].copy()
    standard_auc = get_last_layer_auc(standard, ['model', 'dataset', 'bias'], classifier, probe, step)

    # Cross-model results
    cross = df[df['tag'].str.contains('eval_model=', na=False) &
               ~df['tag'].str.contains('permuted', na=False)].copy()
    cross['eval_model'] = cross['tag'].apply(
        lambda t: parse_cross_tag(t).get('eval_model', ''))
    cross = cross[cross['eval_model'] != '']
    cross_auc = get_last_layer_auc(cross, ['model', 'dataset', 'bias', 'eval_model'], classifier, probe, step)
    cross_auc = cross_auc.rename(columns={'model': 'train_model'})

    # Determine which models have cross-model data
    models_with_data = sorted(set(cross_auc['train_model'].unique()) | set(cross_auc['eval_model'].unique()))
    if not models_with_data:
        return None, []
    # Keep only models in MODEL_ORDER that have data
    model_order = [m for m in MODEL_ORDER if m in models_with_data]

    matrix = np.full((len(model_order), len(model_order)), np.nan)
    for i, train_m in enumerate(model_order):
        # Diagonal
        diag_rows = standard_auc[standard_auc['model'] == train_m]
        if len(diag_rows) > 0:
            matrix[i, i] = diag_rows['auc'].mean()
        # Off-diagonal
        for j, eval_m in enumerate(model_order):
            if i == j:
                continue
            cross_rows = cross_auc[
                (cross_auc['train_model'] == train_m) &
                (cross_auc['eval_model'] == eval_m)
            ]
            if len(cross_rows) > 0:
                matrix[i, j] = cross_rows['auc'].mean()

    return matrix, model_order


def plot_heatmap(matrix, labels, title, filename, label_map=None):
    """Plot a transfer heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    display_labels = [label_map.get(l, l) if label_map else l for l in labels]

    im = ax.imshow(matrix, cmap='YlGn', vmin=0.5, vmax=1.0, aspect='equal', origin='lower')
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
                color = 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=12)

    plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved {FIGURES_DIR / filename}")
    plt.close()


def plot_permutation_comparison(df, classifier='rfm', probe='mot_vs_alg', step=None):
    """Permutation baseline plots in Figure-4 style: 1x2 grid with grouped bars per model.

    Left panel: by dataset, right panel: by hint type.
    Solid bars = real labels, hatched bars = permuted labels.
    """
    MODEL_COLORS = {
        "qwen-3-8b": "#8856a7",
        "llama-3.1-8b": "#e34a33",
        "gemma-3-4b": "#f4c20d",
    }

    # Permuted results
    perm = df[df['tag'].str.contains('permuted_eval', na=False)].copy()
    perm_auc = get_last_layer_auc(perm, ['model', 'dataset', 'bias'], classifier, probe, step)
    perm_auc = perm_auc.rename(columns={'auc': 'perm_auc'})

    # Standard results
    standard = df[df['tag'] == ''].copy()
    std_auc = get_last_layer_auc(standard, ['model', 'dataset', 'bias'], classifier, probe, step)
    std_auc = std_auc.rename(columns={'auc': 'real_auc'})

    if len(perm_auc) == 0:
        print("No permutation results found, skipping plots.")
        return

    # Merge real and permuted
    merged = std_auc.merge(perm_auc, on=['model', 'dataset', 'bias'], how='inner')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = [m for m in MODEL_ORDER if m in merged['model'].unique()]
    datasets = [d for d in DATASET_ORDER if d in merged['dataset'].unique()]
    biases = [b for b in BIAS_ORDER if b in merged['bias'].unique()]
    n_models = len(models)
    width = 0.25

    # --- Combined figure: by dataset (left) and by hint type (right) ---
    fig, (ax_ds, ax_bias) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left panel: by dataset, bars grouped by model
    x_ds = np.arange(len(datasets))
    for idx, model in enumerate(models):
        color = MODEL_COLORS.get(model, '#999999')
        label = MODEL_LABELS.get(model, model)
        offsets = x_ds + (idx - 1) * width
        # Real (solid)
        real_vals = []
        perm_vals = []
        for d in datasets:
            rows = merged[(merged['model'] == model) & (merged['dataset'] == d)]
            real_vals.append(rows['real_auc'].mean() if len(rows) > 0 else 0)
            perm_vals.append(rows['perm_auc'].mean() if len(rows) > 0 else 0)
        ax_ds.bar(offsets, real_vals, width, label=label, color=color, edgecolor='black')
        ax_ds.bar(offsets, perm_vals, width, color=color, edgecolor='black',
                  hatch='///', alpha=0.4, label='_nolegend_')

    ax_ds.set_xticks(x_ds)
    ax_ds.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax_ds.set_ylabel('AUC')
    ax_ds.set_title('Permutation Baseline across Datasets')
    ax_ds.set_ylim(0, 1.05)
    ax_ds.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_ds.grid(axis='y', alpha=0.3)

    # Right panel: by hint type, bars grouped by model
    x_bias = np.arange(len(biases))
    for idx, model in enumerate(models):
        color = MODEL_COLORS.get(model, '#999999')
        offsets = x_bias + (idx - 1) * width
        real_vals = []
        perm_vals = []
        for b in biases:
            rows = merged[(merged['model'] == model) & (merged['bias'] == b)]
            real_vals.append(rows['real_auc'].mean() if len(rows) > 0 else 0)
            perm_vals.append(rows['perm_auc'].mean() if len(rows) > 0 else 0)
        ax_bias.bar(offsets, real_vals, width, color=color, edgecolor='black')
        ax_bias.bar(offsets, perm_vals, width, color=color, edgecolor='black',
                    hatch='///', alpha=0.4, label='_nolegend_')

    ax_bias.set_xticks(x_bias)
    ax_bias.set_xticklabels([BIAS_LABELS.get(b, b) for b in biases])
    ax_bias.set_title('Permutation Baseline across Hint Types')
    ax_bias.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_bias.grid(axis='y', alpha=0.3)

    # Legend: model colors + hatched explanation
    handles, labels = ax_ds.get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor='white', edgecolor='black', hatch='///'))
    labels.append('Permuted')
    ax_ds.legend(handles, labels, loc='lower left', fontsize=11, title='Model')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'permutation_baseline.png', dpi=150, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'permutation_baseline.pdf', bbox_inches='tight')
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
    for step, step_label in [(0, 'Pre-Generation'), (2, 'Post-Generation')]:
        print(f"\n=== Cross-dataset transfer ({step_label}, step {step}) ===")
        xd_matrix = build_cross_dataset_matrix(df, args.classifier, args.probe, step=step)
        print(pd.DataFrame(xd_matrix,
            index=[DATASET_LABELS[d] for d in DATASET_ORDER],
            columns=[DATASET_LABELS[d] for d in DATASET_ORDER]).to_string())
        plot_heatmap(xd_matrix, DATASET_ORDER,
                     f'Cross-Dataset Transfer ({step_label})',
                     f'cross_dataset_transfer_{step_label.lower().replace("-", "_")}.png',
                     DATASET_LABELS)

    # Cross-hint transfer matrix
    for step, step_label in [(0, 'Pre-Generation'), (2, 'Post-Generation')]:
        print(f"\n=== Cross-hint transfer ({step_label}, step {step}) ===")
        xb_matrix = build_cross_hint_matrix(df, args.classifier, args.probe, step=step)
        print(pd.DataFrame(xb_matrix,
            index=[BIAS_LABELS[b] for b in BIAS_ORDER],
            columns=[BIAS_LABELS[b] for b in BIAS_ORDER]).to_string())
        plot_heatmap(xb_matrix, BIAS_ORDER,
                     f'Cross-Hint Transfer ({step_label})',
                     f'cross_hint_transfer_{step_label.lower().replace("-", "_")}.png',
                     BIAS_LABELS)

    # Permutation baseline
    print("\n=== Permutation baseline ===")
    plot_permutation_comparison(df, args.classifier, args.probe, step=2)

    # Cross-model transfer matrix
    for step, step_label in [(0, 'Pre-Generation'), (2, 'Post-Generation')]:
        print(f"\n=== Cross-model transfer ({step_label}, step {step}) ===")
        xm_matrix, xm_models = build_cross_model_matrix(df, args.classifier, args.probe, step=step)
        if xm_matrix is not None and len(xm_models) > 0:
            print(pd.DataFrame(xm_matrix,
                index=[MODEL_LABELS.get(m, m) for m in xm_models],
                columns=[MODEL_LABELS.get(m, m) for m in xm_models]).to_string())
            plot_heatmap(xm_matrix, xm_models,
                         f'Cross-Model Transfer ({step_label})',
                         f'cross_model_transfer_{step_label.lower().replace("-", "_")}.png',
                         MODEL_LABELS)
        else:
            print("  No cross-model results found.")


if __name__ == '__main__':
    main()
