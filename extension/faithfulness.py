import os
os.environ["HF_HOME"] = "/work/hdd/bbjr/pmirtaheri/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

choices = ["A", "B", "C", "D"]

def format_example(example, include_answer=True):
    prompt = f"Question: {example['question']}\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{choices[i]}. {choice}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += f" {example['answer']}\n"
    return prompt


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    torch_dtype=torch.float16,
    device_map="auto")
model.eval()

dataset = load_dataset("cais/mmlu", "all", split="test")

for idx, example in enumerate(dataset):
    prompt = format_example(example, include_answer=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Example {idx + 1}:")
    print(f"Correct Answer: {example['answer']}")
    print(f"{generated_text}")
    print("-" * 80)