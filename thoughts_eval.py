
import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr, contextmanager

import numpy as np
import pandas as pd
import torch
from thoughts.evaluate import evaluate_model
from thoughts.multiple_choice import generate_responses, probe_responses, evaluate_probe, evaluate_responses
from direction_utils import train_rfm_probe_on_concept, accuracy_fn
from utils import split_indices

    

def plot_accuracy_heatmap(df, title, file_name, x_label="Step into Chain of Thought", y_label="Depth in Model", figsize=(10, 6), cmap="viridis", baseline=0.25):
    """
    Plot a heatmap of accuracy from a DataFrame with columns: hidden_layer, gen_step, accuracy.
    
    Parameters:
    - df: pandas DataFrame with 'hidden_layer', 'gen_step', and 'accuracy' columns
    - title: Title of the plot
    - x_label: Label for x-axis (default assumes CoT steps)
    - y_label: Label for y-axis (default assumes model depth)
    - figsize: Tuple for figure size
    - cmap: Color map to use (e.g., 'viridis', 'coolwarm')
    """
    # Round for clean axis labels
    df = df.copy()
    df["layer"] = df["layer"].round(2)
    df["step"] = df["step"].round(2)
    
    # Pivot and sort
    heatmap_data = df.pivot(index="layer", columns="step", values="auc").sort_index(ascending=False)

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_data,
        annot=False,
        fmt=".0f",
        cmap=cmap,
        vmin=baseline,
        vmax=1.0,
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Evaluate language models on various tasks")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str)
    parser.add_argument("--reason_first", action='store_true', help="Use Chain of Thought generation")
    parser.add_argument("--bias", type=str, default=None, help="Bias to apply (e.g., 'expert', 'self')")
    parser.add_argument("--hint_idx", type=int, default=0, help="Index of the hint to provide")
    parser.add_argument("--generate", action='store_true', help="Generate outputs")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate generated outputs")
    parser.add_argument("--probe", type=str, help="Train and evaluate probes") 
    # parser.add_argument("--per_layer", action='store_true', help="Train a probe per layer")    
    parser.add_argument("--universal", action='store_true', help="Use a universal probe")   
    parser.add_argument("--balanced", action='store_true', help="Use balanced examples for probing") 

    parser.add_argument("--n_gen", type=int,  help="Number of responses to generate")
    parser.add_argument("--bs_gen", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--n_train", type=int, help="Number of responses to train on")
    parser.add_argument("--bs_probe", type=int, default=64, help="Batch size for probing")
    parser.add_argument("--n_ckpts", type=int, help="Number of samples from each response")
    parser.add_argument("--ckpt", type=str, default="rel", help="Checkpointing strategy (rel, prefix, suffix)")
    # parser.add_argument("--n_test", type=int,  help="Number of responses to test on")    
    args = parser.parse_args()

    split = args.split or ('train' if args.dataset in ['aqua', 'commonsense_qa'] else 'test')
    reason_first = args.reason_first or (args.bias in ['expert', 'metadata'])

    if args.generate:
        generate_responses(args.model, args.dataset, split, reason_first, args.bias, args.hint_idx, args.n_gen, args.bs_gen)
    if args.evaluate:
        evaluate_responses(args.model, args.dataset, split)    
    if args.probe:
        probe_responses(args.model, args.dataset, split, args.n_train, args.bias, args.probe, args.n_ckpts, args.ckpt, args.universal, args.balanced, args.bs_probe)

