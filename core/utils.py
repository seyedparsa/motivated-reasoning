import json
import torch
import re
import os
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk


# cache_dir = "/work/nvme/bbjr/huggingface/"
# cache_dir = "~/huggingface/"


def get_tokenizer(model_name):
    with open("core/configs/models.json", "r") as f:
        config = json.load(f)
        model_repo = config[model_name]['repo']

    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True, use_fast=True, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


DEFAULT_SAMPLING = {
    "do_sample": True,
    "temperature": 0.1,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
}


def get_sampling_config(model_name):
    with open("core/configs/models.json", "r") as f:
        config = json.load(f)
    return config[model_name].get("sampling", DEFAULT_SAMPLING)


def get_chat_template_kwargs(model_name):
    """Returns extra kwargs to pass to tokenizer.apply_chat_template, from models.json.

    Defaults to {} so the model's default template behavior applies. To disable
    Qwen3 thinking mode, declare `"chat_template_kwargs": {"enable_thinking": false}`
    in models.json for that entry.
    """
    with open("core/configs/models.json", "r") as f:
        config = json.load(f)
    return config[model_name].get("chat_template_kwargs", {})


def get_model(model_name, device="auto"):
    print(f"Loading model: {model_name}")
    with open("core/configs/models.json", "r") as f:
        config = json.load(f)
        model_repo = config[model_name]['repo']

    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True, use_fast=True, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
        # Summarize GPU types, e.g., A100x4, H200x8
        n_gpus = torch.cuda.device_count()
        def _gpu_family(device_name: str) -> str:
            known = [
                "H200", "H100", "A100", "A800", "A10", "A30", "A40", "A5000",
                "V100", "T4", "L4", "L40S", "L40", "RTX 6000", "RTX 5000",
            ]
            for k in known:
                if k in device_name:
                    return k.replace(" ", "")
            m = re.search(r"(H|A|V)\d{2,3}", device_name)
            if m:
                return m.group(0)
            m = re.search(r"RTX\s?\d{3,4}", device_name)
            if m:
                return m.group(0).replace(" ", "")
            # Fallback: use the last token (often model family)
            parts = device_name.split()
            return parts[-1] if parts else device_name
        fams = [_gpu_family(torch.cuda.get_device_name(i)) for i in range(n_gpus)]
        fam_counts = Counter(fams)
        fam_str = ", ".join([f"{k}x{v}" for k, v in sorted(fam_counts.items())])
        print(f"GPUs detected: {fam_str}")
    else:
        dtype = torch.float32
        device_map = "cpu"
        print("CUDA not available, using CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        # cache_dir=cache_dir
    )
    
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None    
    model.eval()
    
    print(f"Model {model_name} loaded successfully.")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    return model, tokenizer

def get_dataset(dataset_name, split=None, max_size=None, start_size=0):
    print(f"Loading dataset: {dataset_name}")
    with open("core/configs/datasets.json", "r") as f:
        config = json.load(f)
        entry = config[dataset_name]
        repo = entry['repo']
        subset = entry.get('subset', None)
        if split is None:
            split = entry.get('split', None)
            print(f"Using default split: {split}")
    if subset:
        dataset = load_dataset(repo, subset, split=split)
    else:
        dataset = load_dataset(repo, split=split)
    # Optional slicing: rows [start_size, max_size). When start_size=0 this
    # matches the historical behaviour (first max_size rows).
    if max_size is None:
        max_size = len(dataset)
    if start_size or max_size < len(dataset):
        end = min(max_size, len(dataset))
        start = min(start_size, end)
        dataset = dataset.select(range(start, end))
    print(f"Dataset {dataset_name} loaded successfully.")
    print(f"Dataset size: {len(dataset)}")
    return dataset


def get_choices(dataset_name):
    """Returns the number of choices for the given dataset."""
    if dataset_name in ['mmlu', "arc-challenge"]:
        valid_choices = [f"{chr(65 + i)}" for i in range(4)]
    elif dataset_name in ['aqua', 'commonsense_qa']:
        valid_choices = [f"{chr(65 + i)}" for i in range(5)]
    elif dataset_name in ['bbh-causal_judgement']:
        valid_choices = ['Yes', 'No']
    elif dataset_name in ['bbh-formal_fallacies']:
        valid_choices = ['valid', 'invalid']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return valid_choices