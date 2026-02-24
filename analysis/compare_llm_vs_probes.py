#!/usr/bin/env python3
"""Compare LLM baseline vs probe performance for each (model, dataset, bias) combination."""

import os
import sqlite3
import pandas as pd

MOTIVATION_HOME = os.getenv("MOTIVATION_HOME", "outputs")
PROBE_DB = os.path.join(MOTIVATION_HOME, "probe_metrics.db")
LLM_DB = os.path.join(MOTIVATION_HOME, "llm_metrics.db")


def load_probe_metrics(strategy="max_layer_step"):
    """Load probe metrics with different selection strategies.

    Pivots per-classifier rows back to wide format (rfm_accuracy, rfm_auc,
    linear_accuracy, linear_auc) for comparison.

    Args:
        strategy: "max_layer_step" - take max layer and step for each group
                  "max_rfm_auc" - take row with highest RFM AUC for each group
    """
    conn = sqlite3.connect(PROBE_DB)

    if strategy == "max_rfm_auc":
        query = """
        SELECT
            model, dataset, bias,
            layer, step, classifier,
            accuracy, auc
        FROM probe_metrics p1
        WHERE classifier = 'rfm' AND auc = (
            SELECT MAX(auc) FROM probe_metrics p2
            WHERE p2.classifier = 'rfm'
            AND p2.model = p1.model
            AND p2.dataset = p1.dataset
            AND p2.bias = p1.bias
        )
        GROUP BY model, dataset, bias
        """
        rfm_df = pd.read_sql_query(query, conn)
        rfm_df = rfm_df.rename(columns={"accuracy": "rfm_accuracy", "auc": "rfm_auc"})
        rfm_df = rfm_df.drop(columns=["classifier"], errors="ignore")

        # Get matching linear rows at same layer/step
        linear_df = pd.read_sql_query(
            "SELECT model, dataset, bias, layer, step, accuracy, auc FROM probe_metrics WHERE classifier = 'linear'",
            conn,
        )
        linear_df = linear_df.rename(columns={"accuracy": "linear_accuracy", "auc": "linear_auc"})
        df = pd.merge(rfm_df, linear_df, on=["model", "dataset", "bias", "layer", "step"], how="left")
    else:  # max_layer_step
        query = """
        SELECT
            model, dataset, bias,
            layer, step, classifier,
            accuracy, auc
        FROM probe_metrics
        WHERE layer = (
            SELECT MAX(layer) FROM probe_metrics p2
            WHERE p2.model = probe_metrics.model
            AND p2.dataset = probe_metrics.dataset
            AND p2.bias = probe_metrics.bias
        )
        AND step = (
            SELECT MAX(step) FROM probe_metrics p3
            WHERE p3.model = probe_metrics.model
            AND p3.dataset = probe_metrics.dataset
            AND p3.bias = probe_metrics.bias
            AND p3.layer = probe_metrics.layer
        )
        """
        raw = pd.read_sql_query(query, conn)
        rfm = raw[raw["classifier"] == "rfm"].rename(columns={"accuracy": "rfm_accuracy", "auc": "rfm_auc"})
        linear = raw[raw["classifier"] == "linear"].rename(columns={"accuracy": "linear_accuracy", "auc": "linear_auc"})
        key_cols = ["model", "dataset", "bias", "layer", "step"]
        rfm = rfm.drop(columns=["classifier"], errors="ignore")
        linear = linear.drop(columns=["classifier"], errors="ignore")
        df = pd.merge(rfm, linear, on=key_cols, how="outer")

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


def compare_metrics(strategy="max_layer_step"):
    """Compare LLM vs probe performance."""
    probe_df = load_probe_metrics(strategy=strategy)
    llm_df = load_llm_metrics()

    # Merge on (model, dataset, bias)
    merged = pd.merge(
        llm_df, probe_df,
        on=["model", "dataset", "bias"],
        how="outer",
        suffixes=("_llm", "_probe")
    )

    # Calculate differences (probe - llm, positive means probe is better)
    merged["rfm_vs_llm_acc"] = merged["rfm_accuracy"] - merged["llm_accuracy"]
    merged["rfm_vs_llm_auc"] = (merged["rfm_auc"] - merged["llm_auc"]) * 100  # Convert to percentage points
    merged["linear_vs_llm_acc"] = merged["linear_accuracy"] - merged["llm_accuracy"]
    merged["linear_vs_llm_auc"] = (merged["linear_auc"] - merged["llm_auc"]) * 100

    return merged


def print_comparison(df, strategy="max_layer_step"):
    """Print a formatted comparison table."""
    strategy_desc = "max layer/step" if strategy == "max_layer_step" else "best RFM AUC"
    print("\n" + "="*100)
    print(f"LLM vs Probe Comparison ({strategy_desc} per group)")
    print("="*100)

    # Sort by model, dataset, bias
    df = df.sort_values(["model", "dataset", "bias"])

    print(f"\n{'Model':<15} {'Dataset':<15} {'Bias':<10} | {'LLM Acc':>8} {'RFM Acc':>8} {'Lin Acc':>8} | {'LLM AUC':>8} {'RFM AUC':>8} {'Lin AUC':>8}")
    print("-"*100)

    for _, row in df.iterrows():
        model = row["model"][:14] if pd.notna(row["model"]) else "N/A"
        dataset = row["dataset"][:14] if pd.notna(row["dataset"]) else "N/A"
        bias = row["bias"][:9] if pd.notna(row["bias"]) else "N/A"

        llm_acc = f"{row['llm_accuracy']:.1f}" if pd.notna(row["llm_accuracy"]) else "N/A"
        rfm_acc = f"{row['rfm_accuracy']:.1f}" if pd.notna(row["rfm_accuracy"]) else "N/A"
        lin_acc = f"{row['linear_accuracy']:.1f}" if pd.notna(row["linear_accuracy"]) else "N/A"

        llm_auc = f"{row['llm_auc']:.3f}" if pd.notna(row["llm_auc"]) else "N/A"
        rfm_auc = f"{row['rfm_auc']:.3f}" if pd.notna(row["rfm_auc"]) else "N/A"
        lin_auc = f"{row['linear_auc']:.3f}" if pd.notna(row["linear_auc"]) else "N/A"

        print(f"{model:<15} {dataset:<15} {bias:<10} | {llm_acc:>8} {rfm_acc:>8} {lin_acc:>8} | {llm_auc:>8} {rfm_auc:>8} {lin_auc:>8}")

    print("\n" + "="*100)
    print("Summary Statistics (Probe - LLM, positive = probe better)")
    print("="*100)

    # Summary stats
    for metric, col in [("RFM Accuracy", "rfm_vs_llm_acc"),
                        ("Linear Accuracy", "linear_vs_llm_acc"),
                        ("RFM AUC (pp)", "rfm_vs_llm_auc"),
                        ("Linear AUC (pp)", "linear_vs_llm_auc")]:
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"{metric:<20}: mean={valid.mean():+.2f}, min={valid.min():+.2f}, max={valid.max():+.2f}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare LLM vs probe performance")
    parser.add_argument("--strategy", type=str, default="max_layer_step",
                        choices=["max_layer_step", "max_rfm_auc"],
                        help="How to select probe result per group")
    args = parser.parse_args()

    df = compare_metrics(strategy=args.strategy)
    print_comparison(df, strategy=args.strategy)

    # Save to CSV
    suffix = "_best_auc" if args.strategy == "max_rfm_auc" else ""
    output_path = os.path.join(MOTIVATION_HOME, f"llm_vs_probe_comparison{suffix}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")
