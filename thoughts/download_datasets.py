
import os
import datasets

def download_mmlu():
    """Downloads the MMLU dataset from Hugging Face and saves it to the data directory."""
    print("Downloading MMLU dataset...")
    mmlu = datasets.load_dataset("cais/mmlu", "all")
    mmlu.save_to_disk("thoughts/data/mmlu")
    print("MMLU dataset downloaded and saved to thoughts/data/mmlu.")

def download_gpqa():
    """Downloads the GPQA diamond dataset from Hugging Face and saves it to the data directory."""
    print("Downloading GPQA diamond dataset...")
    gpqa = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    gpqa.save_to_disk("thoughts/data/gpqa")
    print("GPQA diamond dataset downloaded and saved to thoughts/data/gpqa.")


def download_bbh(subset="causal_judgement"):
    """Downloads the BBH dataset from Hugging Face and saves it to the data directory."""
    print("Downloading BBH dataset...")
    bbh = datasets.load_dataset("maveriq/bigbenchhard", subset)
    bbh.save_to_disk("thoughts/data/bbh-" + subset)
    print("BBH dataset downloaded and saved to thoughts/data/bbh-" + subset + ".")


def download_arc():
    """Downloads the ARC dataset from Hugging Face and saves it to the data directory."""
    print("Downloading ARC dataset...")
    arc = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")
    arc.save_to_disk("thoughts/data/arc")
    print("ARC dataset downloaded and saved to thoughts/data/arc.")


def download_aqua():
    """Downloads the AQUA dataset from Hugging Face and saves it to the data directory."""
    print("Downloading AQUA dataset...")
    aqua = datasets.load_dataset("deepmind/aqua_rat", "raw")
    aqua.save_to_disk("thoughts/data/aqua")
    print("AQUA dataset downloaded and saved to thoughts/data/aqua.")


def download_cqa():
    """Downloads the CQA dataset from Hugging Face and saves it to the data directory."""
    print("Downloading CQA dataset...")
    cqa = datasets.load_dataset("tau/commonsense_qa")
    cqa.save_to_disk("thoughts/data/cqa")
    print("CQA dataset downloaded and saved to thoughts/data/cqa.")


def download_math():
    """Downloads the MATH dataset from Hugging Face and saves it to the data directory."""
    print("Downloading MATH dataset...")
    math_dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")
    math_dataset.save_to_disk("thoughts/data/math")
    print("MATH dataset downloaded and saved to thoughts/data/math.")

def download_gsm8k():
    """Downloads the GSM8K dataset from Hugging Face and saves it to the data directory."""
    print("Downloading GSM8K dataset...")
    gsm8k = datasets.load_dataset("gsm8k", "main")
    gsm8k.save_to_disk("thoughts/data/gsm8k")
    print("GSM8K dataset downloaded and saved to thoughts/data/gsm8k.")

if __name__ == "__main__":
    if not os.path.exists("thoughts/data/mmlu"):
        download_mmlu()
    else:
        print("MMLU dataset already exists in thoughts/data/mmlu.")

    if not os.path.exists("thoughts/data/gpqa"):
        download_gpqa()
    else:
        print("GPQA dataset already exists in thoughts/data/gpqa.")

    if not os.path.exists("thoughts/data/math"):
        download_math()
    else:
        print("MATH dataset already exists in thoughts/data/math.")

    if not os.path.exists("thoughts/data/gsm8k"):
        download_gsm8k()
    else:
        print("GSM8K dataset already exists in thoughts/data/gsm8k.")
    
    if not os.path.exists("thoughts/data/bbh-causal_judgement"):
        download_bbh("causal_judgement")
    else:
        print("BBH dataset already exists in thoughts/data/bbh-causal_judgement.")

    if not os.path.exists("thoughts/data/bbh-formal_fallacies"):
        download_bbh("formal_fallacies")
    else:
        print("BBH formal fallacies dataset already exists in thoughts/data/bbh-formal_fallacies.")

    if not os.path.exists("thoughts/data/arc"):
        download_arc()
    else:
        print("ARC dataset already exists in thoughts/data/arc.")

    if not os.path.exists("thoughts/data/aqua"):
        download_aqua()
    else:
        print("AQUA dataset already exists in thoughts/data/aqua.")

    if not os.path.exists("thoughts/data/cqa"):     
        download_cqa()
    else:
        print("CQA dataset already exists in thoughts/data/cqa.")

    print("All datasets checked/updated.")