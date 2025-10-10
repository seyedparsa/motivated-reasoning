import torch
import json
import os
from datetime import datetime
from datasets import load_from_disk
from tqdm import tqdm
from thoughts.utils import get_model
import numpy as np
import re
import random
from transformers import StoppingCriteria, StoppingCriteriaList

class FinalAnswerStoppingCriteria(StoppingCriteria):
    """Stop generation when 'Final answer:' followed by a letter is generated."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.final_answer_pattern = re.compile(r'Final answer:\s*([ABCD])', re.IGNORECASE)
    
    def __call__(self, input_ids, scores, **kwargs):
        # Get the last generated tokens
        for batch_idx, sequence in enumerate(input_ids):
            # Decode the sequence to text
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            
            # Check if we have a final answer
            if self.final_answer_pattern.search(text):
                return True
        
        return False

def generate_answer_no_cot(model, tokenizer, prompt):
    """Generate answer without chain of thought - single token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": 1,  # Single token only
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        outputs = model.generate(**generation_kwargs)
    
    # Extract the generated response
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Return the single character
    return generated_text.strip().upper(), generated_text.strip()

def generate_answer_cot(model, tokenizer, prompt):
    """Generate answer with chain of thought reasoning."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create stopping criteria
    stopping_criteria = StoppingCriteriaList([FinalAnswerStoppingCriteria(tokenizer)])
    
    with torch.no_grad():
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": 2048,  # Reduced from 2048 for speed
            "do_sample": False,  # Changed to False for speed
            "pad_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        
        outputs = model.generate(**generation_kwargs)
    
    # Extract the generated response
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract the final answer from the response
    match = re.search(r'Final answer:\s*([ABCD])', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), generated_text.strip()
    
    # If no clear answer found, return empty and the full text
    return "", generated_text.strip()

def evaluate_multiple_choice(model, tokenizer, dataset, dataset_name):
    """Evaluate model on multiple choice datasets with both No-CoT and CoT approaches."""
    # Track results for both approaches
    no_cot_correct = 0
    cot_correct = 0
    total = 0
    
    # Track detailed comparison (6 categories)
    comparison_counts = {
        "cc": 0,  # Both correct (correct to correct)
        "c2w": 0, # No-CoT correct, CoT wrong (correct to wrong)
        "c2e": 0, # No-CoT correct, CoT empty (correct to empty)
        "w2c": 0, # No-CoT wrong, CoT correct (wrong to correct)
        "ww": 0,  # Both wrong (wrong to wrong)
        "w2e": 0, # No-CoT wrong, CoT empty (wrong to empty)
    }
    
    # Determine which split to use and data format based on dataset
    if dataset_name == "mmlu":
        test_data = dataset['test']
        log_interval = 100
    elif dataset_name == "gpqa":
        test_data = dataset['train']  # GPQA only has 'train' split
        log_interval = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    for example in tqdm(test_data):
        # Extract question and choices based on dataset format
        if dataset_name == "mmlu":
            question = example['question']
            choices = example['choices']
            correct_answer_idx = example['answer']
        elif dataset_name == "gpqa":
            question = example['Question']  # Note: Capital Q
            choices = [example['Incorrect Answer 1'], example['Incorrect Answer 2'], 
                      example['Incorrect Answer 3'], example['Correct Answer']]
            correct_answer_idx = 3  # Correct answer is always the last one in GPQA
            
            # Shuffle choices for GPQA and track correct answer position
            choice_indices = list(range(4))
            random.shuffle(choice_indices)
            choices = [choices[i] for i in choice_indices]
            correct_answer_idx = choice_indices.index(3)
        
        # Create base question format
        base_prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            base_prompt += f"{chr(65+i)}. {choice}\n"
        
        # No-CoT prompt: Direct answer request with explicit instruction
        no_cot_prompt = base_prompt + "Answer with a single letter (A, B, C, or D):"
        
        # CoT prompt: Request reasoning
        cot_prompt = base_prompt + "Think step by step. Then, based on your reasoning, provide the single most likely answer choice in the format Final answer: X, where X is A, B, C, or D."
        
        # Generate answers with both approaches
        no_cot_answer, no_cot_response = generate_answer_no_cot(model, tokenizer, no_cot_prompt)
        cot_answer, cot_response = generate_answer_cot(model, tokenizer, cot_prompt)
        
        # Determine correctness for each approach
        no_cot_is_correct = False
        cot_is_correct = False
        cot_is_empty = False
        
        # Check No-CoT correctness
        if no_cot_answer in ['A', 'B', 'C', 'D']:
            predicted_idx = ord(no_cot_answer) - ord('A')
            if predicted_idx == correct_answer_idx:
                no_cot_correct += 1
                no_cot_is_correct = True
        
        # Check CoT correctness and emptiness
        if cot_answer == "":
            cot_is_empty = True
        elif cot_answer in ['A', 'B', 'C', 'D']:
            predicted_idx = ord(cot_answer) - ord('A')
            if predicted_idx == correct_answer_idx:
                cot_correct += 1
                cot_is_correct = True
        
        # Update detailed comparison counts
        if no_cot_is_correct and cot_is_correct:
            comparison_counts["cc"] += 1
        elif no_cot_is_correct and not cot_is_correct and not cot_is_empty:
            comparison_counts["c2w"] += 1
        elif no_cot_is_correct and cot_is_empty:
            comparison_counts["c2e"] += 1
        elif not no_cot_is_correct and cot_is_correct:
            comparison_counts["w2c"] += 1
        elif not no_cot_is_correct and not cot_is_correct and not cot_is_empty:
            comparison_counts["ww"] += 1
        elif not no_cot_is_correct and cot_is_empty:
            comparison_counts["w2e"] += 1
        
        total += 1
        
        if total % log_interval == 0:
            correct_choice = chr(65 + correct_answer_idx)
            no_cot_accuracy = no_cot_correct / total
            cot_accuracy = cot_correct / total
            
            print(f"CoT response: {cot_response}")
            print(f"Processed {total} questions")
            print(f"No-CoT accuracy: {no_cot_accuracy:.3f} ({no_cot_correct}/{total})")
            print(f"CoT accuracy: {cot_accuracy:.3f} ({cot_correct}/{total})")
            print(f"Correct answer: '{correct_choice}'")
            print(f"No-CoT answer: '{no_cot_answer}' | CoT answer: '{cot_answer}'")
            print("-" * 80)
    
    no_cot_accuracy = no_cot_correct / total
    cot_accuracy = cot_correct / total
    
    return {
        "no_cot": {"accuracy": no_cot_accuracy, "correct": no_cot_correct, "total": total},
        "cot": {"accuracy": cot_accuracy, "correct": cot_correct, "total": total},
        "total_questions": total,
        "detailed_breakdown": comparison_counts
    }

def evaluate_model(model_name, dataset_name):
    """Evaluates a model on a given dataset."""
    print(f"Evaluating model {model_name} on dataset {dataset_name}...")

    # Load model and tokenizer
    model, tokenizer = get_model(model_name)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    # Load dataset
    dataset = load_from_disk(f"thoughts/data/{dataset_name}")

    # Run evaluation based on dataset
    if dataset_name in ["mmlu", "gpqa"]:
        results = evaluate_multiple_choice(model, tokenizer, dataset, dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Evaluation of {model_name} on {dataset_name} complete.")
    print(f"\n=== FINAL RESULTS ===")
    print(f"No-CoT: {results['no_cot']['accuracy']:.3f} ({results['no_cot']['correct']}/{results['no_cot']['total']})")
    print(f"CoT: {results['cot']['accuracy']:.3f} ({results['cot']['correct']}/{results['cot']['total']})")
    print(f"CoT improvement: {results['cot']['accuracy'] - results['no_cot']['accuracy']:.3f}")
    print(f"Total questions: {results['total_questions']}")
    
    # Display detailed breakdown
    print(f"\n=== DETAILED BREAKDOWN ===")
    breakdown = results['detailed_breakdown']
    print(f"Both correct (cc): {breakdown['cc']}")
    print(f"No-CoT correct, CoT incorrect (c2w): {breakdown['c2w']}")
    print(f"No-CoT correct, CoT empty (c2e): {breakdown['c2e']}")
    print(f"No-CoT incorrect, CoT correct (w2c): {breakdown['w2c']}")
    print(f"Both incorrect (ww): {breakdown['ww']}")
    print(f"No-CoT incorrect, CoT empty (w2e): {breakdown['w2e']}")
    
    # Verify totals
    total_breakdown = sum(breakdown.values())
    print(f"Total from breakdown: {total_breakdown} (should equal {results['total_questions']})")
    
    # Show percentages
    print(f"\n=== PERCENTAGES ===")
    for key, count in breakdown.items():
        percentage = (count / results['total_questions']) * 100
        print(f"{key}: {percentage:.1f}% ({count}/{results['total_questions']})")
    
    # CoT improvement analysis
    cot_helps = breakdown['w2c']
    cot_hurts = breakdown['c2w']
    print(f"\n=== COT IMPACT ANALYSIS ===")
    print(f"CoT helps (incorrect → correct): {cot_helps} ({cot_helps/results['total_questions']*100:.1f}%)")
    print(f"CoT hurts (correct → incorrect): {cot_hurts} ({cot_hurts/results['total_questions']*100:.1f}%)")
    print(f"Net CoT benefit: {cot_helps - cot_hurts} questions ({(cot_helps - cot_hurts)/results['total_questions']*100:.1f}%)")

    # Save results
    save_results(model_name, dataset_name, results)
    
    return results


def save_results(model_name, dataset_name, results):
    """Save evaluation results to JSON files for both approaches."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = model_name.replace("/", "_")
    
    # Save No-CoT results
    no_cot_filename = f"thoughts/results/no-CoT-{model_clean}_{dataset_name}_{timestamp}.json"
    no_cot_data = {
        "model": model_name,
        "dataset": dataset_name,
        "approach": "no-CoT",
        "timestamp": timestamp,
        "results": results["no_cot"],
        "detailed_breakdown": results["detailed_breakdown"]
    }
    
    # Save CoT results
    cot_filename = f"thoughts/results/CoT-{model_clean}_{dataset_name}_{timestamp}.json"
    cot_data = {
        "model": model_name,
        "dataset": dataset_name,
        "approach": "CoT",
        "timestamp": timestamp,
        "results": results["cot"],
        "detailed_breakdown": results["detailed_breakdown"]
    }
    
    # Save combined results
    combined_filename = f"thoughts/results/combined-{model_clean}_{dataset_name}_{timestamp}.json"
    combined_data = {
        "model": model_name,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "results": results
    }
    
    os.makedirs("results", exist_ok=True)
    
    with open(no_cot_filename, 'w') as f:
        json.dump(no_cot_data, f, indent=2)
    
    with open(cot_filename, 'w') as f:
        json.dump(cot_data, f, indent=2)
        
    with open(combined_filename, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"No-CoT results saved to {no_cot_filename}")
    print(f"CoT results saved to {cot_filename}")
    print(f"Combined results saved to {combined_filename}")
