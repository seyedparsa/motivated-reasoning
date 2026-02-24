#!/usr/bin/env python3
"""Plot RFM accuracy and AUC heatmaps for each (model, dataset, bias) combination."""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MOTIVATION_HOME = os.getenv("MOTIVATION_HOME", "outputs")
PROBE_DB = os.path.join(MOTIVATION_HOME, "probe_metrics.db")
LLM_DB = os.path.join(MOTIVATION_HOME, "llm_metrics.db")
OUTPUT_DIR = os.path.join(MOTIVATION_HOME, "probe_grids")

STEP_LABELS = ['Beginning', 'Middle', 'End']


def load_all_probe_metrics():
    """Load all probe metrics."""
    conn = sqlite3.connect(PROBE_DB)
    query = """
    SELECT
        model, dataset, bias, probe, classifier,
        layer, step,
        test_examples,
        n_zeros, n_ones,
        accuracy, auc
    FROM probe_metrics
    WHERE typeof(layer) = 'integer' AND typeof(step) = 'integer' AND n_ckpts = 3
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_llm_metrics():
    """Load LLM baseline metrics."""
    conn = sqlite3.connect(LLM_DB)
    query = """
    SELECT
        model, dataset, bias,
        llm_accuracy, llm_auc
    FROM llm_metrics
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def plot_heatmap(data, title, output_path, metric_name, vmin=None, vmax=None,
                 test_examples=None, n_zeros=None, n_ones=None, llm_value=None, llm_label=None):
    """Plot a single heatmap."""
    # Ensure layer and step are integers
    data = data.copy()
    data['layer'] = data['layer'].astype(int)
    data['step'] = data['step'].astype(int)
    # Deduplicate by (layer, step), keeping last occurrence
    data = data.drop_duplicates(subset=['layer', 'step'], keep='last')
    # Get unique sorted layers and steps
    layers = sorted(data['layer'].unique())
    steps = sorted(data['step'].unique())

    # Create pivot table
    pivot = data.pivot(index='layer', columns='step', values=metric_name)

    # Reindex to ensure proper ordering (layers descending so higher layers are at top)
    sorted_layers = sorted(layers, reverse=True)
    pivot = pivot.reindex(index=sorted_layers, columns=steps)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

    # Set x-axis ticks - use Beginning, Middle, End
    ax.set_xticks(range(len(steps)))
    step_labels = STEP_LABELS[:len(steps)] if len(steps) <= len(STEP_LABELS) else [f"Step {i+1}" for i in range(len(steps))]
    ax.set_xticklabels(step_labels)

    # Set y-axis ticks - use actual layer values
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in sorted_layers])

    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")

    # Build title with test examples, distribution, and LLM baseline
    title_parts = [title]
    subtitle_parts = []
    if test_examples is not None:
        if n_zeros is not None and n_ones is not None:
            subtitle_parts.append(f"n={test_examples} [{n_zeros}:{n_ones}]")
        else:
            subtitle_parts.append(f"n={test_examples}")
    if llm_value is not None and llm_label is not None:
        subtitle_parts.append(f"LLM {llm_label}={llm_value:.1f}" if llm_label == "Acc" else f"LLM {llm_label}={llm_value:.3f}")
    if subtitle_parts:
        title_parts.append(f"({', '.join(subtitle_parts)})")
    ax.set_title('\n'.join(title_parts))

    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax)
    if 'auc' in metric_name.lower():
        cbar.set_label('AUC')
    else:
        cbar.set_label('Accuracy (%)')

    # Add value annotations
    for i in range(len(layers)):
        for j in range(len(steps)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
                if 'auc' in metric_name.lower():
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=text_color, fontsize=8)
                else:
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_all_probe_metrics()
    llm_df = load_llm_metrics()

    # Create lookup dict for LLM metrics
    llm_lookup = {}
    for _, row in llm_df.iterrows():
        key = (row['model'], row['dataset'], row['bias'])
        llm_lookup[key] = {'accuracy': row['llm_accuracy'], 'auc': row['llm_auc']}

    # Group by (model, dataset, bias, probe)
    groups = df.groupby(['model', 'dataset', 'bias', 'probe'])

    for (model, dataset, bias, probe), group_df in groups:
        # Get test_examples and distribution (should be same for all rows in group)
        test_examples = group_df['test_examples'].iloc[0] if 'test_examples' in group_df.columns else None
        n_zeros = group_df['n_zeros'].iloc[0] if 'n_zeros' in group_df.columns and pd.notna(group_df['n_zeros'].iloc[0]) else None
        n_ones = group_df['n_ones'].iloc[0] if 'n_ones' in group_df.columns and pd.notna(group_df['n_ones'].iloc[0]) else None

        # Get LLM baseline metrics
        llm_metrics = llm_lookup.get((model, dataset, bias), {})
        llm_acc = llm_metrics.get('accuracy')
        llm_auc = llm_metrics.get('auc')

        print(f"Plotting {model} / {dataset} / {bias} / {probe} (n={test_examples} [{n_zeros}:{n_ones}], LLM acc={llm_acc}, auc={llm_auc})...")

        # Clean name for filename
        name = f"{model}_{dataset}_{bias}_{probe}"

        for clf_name, clf_df in group_df.groupby("classifier"):
            clf_label = clf_name.upper()

            # Plot Accuracy
            acc_path = os.path.join(OUTPUT_DIR, f"{name}_{clf_name}_accuracy.png")
            plot_heatmap(
                clf_df,
                f"{clf_label} Accuracy: {model} / {dataset} / {bias} / {probe}",
                acc_path,
                'accuracy',
                vmin=50, vmax=100,
                test_examples=test_examples,
                n_zeros=n_zeros,
                n_ones=n_ones,
                llm_value=llm_acc,
                llm_label="Acc"
            )

            # Plot AUC
            auc_path = os.path.join(OUTPUT_DIR, f"{name}_{clf_name}_auc.png")
            plot_heatmap(
                clf_df,
                f"{clf_label} AUC: {model} / {dataset} / {bias} / {probe}",
                auc_path,
                'auc',
                vmin=0.5, vmax=1.0,
                test_examples=test_examples,
                n_zeros=n_zeros,
                n_ones=n_ones,
                llm_value=llm_auc,
                llm_label="AUC"
            )

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
