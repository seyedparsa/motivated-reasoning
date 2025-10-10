#!/usr/bin/env python3
"""
Script to analyze evaluation results and display detailed breakdowns.
"""

import json
import os
import glob
from pathlib import Path

def analyze_combined_results(results_file):
    """Analyze a combined results file and display detailed breakdown."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== ANALYSIS OF {Path(results_file).name} ===")
    print(f"Model: {data['model']}")
    print(f"Dataset: {data['dataset']}")
    print(f"Timestamp: {data['timestamp']}")
    
    results = data['results']
    
    # Basic accuracy results
    print(f"\n=== BASIC RESULTS ===")
    print(f"No-CoT: {results['no_cot']['accuracy']:.3f} ({results['no_cot']['correct']}/{results['no_cot']['total']})")
    print(f"CoT: {results['cot']['accuracy']:.3f} ({results['cot']['correct']}/{results['cot']['total']})")
    print(f"CoT improvement: {results['cot']['accuracy'] - results['no_cot']['accuracy']:.3f}")
    print(f"Total questions: {results['total_questions']}")
    
    # Detailed breakdown (if available)
    if 'detailed_breakdown' in results:
        breakdown = results['detailed_breakdown']
        print(f"\n=== DETAILED BREAKDOWN ===")
        print(f"Both correct: {breakdown['no_cot_correct_cot_correct']}")
        print(f"No-CoT correct, CoT incorrect: {breakdown['no_cot_correct_cot_incorrect']}")
        print(f"No-CoT correct, CoT empty: {breakdown['no_cot_correct_cot_empty']}")
        print(f"No-CoT incorrect, CoT correct: {breakdown['no_cot_incorrect_cot_correct']}")
        print(f"Both incorrect: {breakdown['no_cot_incorrect_cot_incorrect']}")
        print(f"No-CoT incorrect, CoT empty: {breakdown['no_cot_incorrect_cot_empty']}")
        
        # Verify totals
        total_breakdown = sum(breakdown.values())
        print(f"Total from breakdown: {total_breakdown} (should equal {results['total_questions']})")
        
        # Show percentages
        print(f"\n=== PERCENTAGES ===")
        for key, count in breakdown.items():
            percentage = (count / results['total_questions']) * 100
            print(f"{key}: {percentage:.1f}% ({count}/{results['total_questions']})")
        
        # CoT improvement analysis
        cot_helps = breakdown['no_cot_incorrect_cot_correct']
        cot_hurts = breakdown['no_cot_correct_cot_incorrect']
        print(f"\n=== COT IMPACT ANALYSIS ===")
        print(f"CoT helps (incorrect → correct): {cot_helps} ({cot_helps/results['total_questions']*100:.1f}%)")
        print(f"CoT hurts (correct → incorrect): {cot_hurts} ({cot_hurts/results['total_questions']*100:.1f}%)")
        print(f"Net CoT benefit: {cot_helps - cot_hurts} questions ({(cot_helps - cot_hurts)/results['total_questions']*100:.1f}%)")
    else:
        print("\n=== NO DETAILED BREAKDOWN AVAILABLE ===")
        print("This appears to be an older results file without detailed breakdown tracking.")

def compare_models(results_files):
    """Compare multiple models/datasets."""
    print(f"\n=== MODEL COMPARISON ===")
    print(f"{'Model':<30} {'Dataset':<10} {'No-CoT':<10} {'CoT':<10} {'Improvement':<12}")
    print("-" * 75)
    
    for file in results_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        model_name = data['model'].split('/')[-1]  # Get just the model name
        dataset = data['dataset']
        results = data['results']
        
        no_cot_acc = results['no_cot']['accuracy']
        cot_acc = results['cot']['accuracy']
        improvement = cot_acc - no_cot_acc
        
        print(f"{model_name:<30} {dataset:<10} {no_cot_acc:<10.3f} {cot_acc:<10.3f} {improvement:<12.3f}")

def main():
    """Main analysis function."""
    results_dir = Path("thoughts/results")
    
    # Find all combined results files
    combined_files = list(results_dir.glob("combined-*.json"))
    
    if not combined_files:
        print("No combined results files found in results/ directory")
        return
    
    # Sort by modification time (newest first)
    combined_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"Found {len(combined_files)} combined results files")
    
    # Analyze each file
    for file in combined_files:
        analyze_combined_results(file)
    
    # Compare all models if we have multiple files
    if len(combined_files) > 1:
        compare_models(combined_files)

if __name__ == "__main__":
    main()
