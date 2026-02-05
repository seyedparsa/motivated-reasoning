#!/bin/bash
# Script to create an anonymized copy of the codebase for conference submission
# Usage: ./create_anonymized_submission.sh [output_directory]

set -e

# Configuration
OUTPUT_DIR="${1:-anonymized_submission}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating anonymized submission in: $OUTPUT_DIR"

# Remove existing output directory if it exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing existing output directory..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# STEP 1: Copy files, excluding sensitive directories and files
# ============================================================================
echo "Copying files (excluding sensitive content)..."

# Use rsync to copy while excluding certain patterns
rsync -av --progress "$SOURCE_DIR/" "$OUTPUT_DIR/" \
    --exclude='.git' \
    --exclude='.git/' \
    --exclude='.github' \
    --exclude='.github/' \
    --exclude='.vscode' \
    --exclude='.vscode/' \
    --exclude='.claude' \
    --exclude='.claude/' \
    --exclude='__pycache__' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='core' \
    --exclude='*.err' \
    --exclude='*.out' \
    --exclude='main.pdf' \
    --exclude='*.png' \
    --exclude='*.jpg' \
    --exclude='*.jpeg' \
    --exclude='simons.yml' \
    --exclude='simons_local.yml' \
    --exclude='outputs/' \
    --exclude='old_outputs/' \
    --exclude='figures/' \
    --exclude='directions/' \
    --exclude='data/' \
    --exclude='create_anonymized_submission.sh' \
    --exclude='CLAUDE.md' \
    --exclude='AGENTS.md'

# ============================================================================
# STEP 2: Anonymize Python files - replace hardcoded paths
# ============================================================================
echo "Anonymizing Python files..."

# Find all Python files and replace identifying information
find "$OUTPUT_DIR" -name "*.py" -type f | while read -r file; do
    # Replace hardcoded user paths
    sed -i 's|/u/dbeaglehole/[^"'\'']*|/path/to/project|g' "$file"
    sed -i 's|/u/pmirtaheri/[^"'\'']*|/path/to/project|g' "$file"
    sed -i 's|/work/hdd/bbjr/pmirtaheri/[^"'\'']*|/path/to/data|g' "$file"

    # Replace HuggingFace usernames
    sed -i 's|seyedparsa/|anonymous-user/|g' "$file"
    sed -i 's|dmbeaglehole/|anonymous-user/|g' "$file"

    # Replace GitHub references
    sed -i 's|github.com/seyedparsa/|github.com/anonymous/|g' "$file"
    sed -i 's|github.com/dmbeaglehole/|github.com/anonymous/|g' "$file"
done

# ============================================================================
# STEP 3: Anonymize shell scripts
# ============================================================================
echo "Anonymizing shell scripts..."

find "$OUTPUT_DIR" -name "*.sh" -type f | while read -r file; do
    # Replace user-specific squeue commands
    sed -i 's|squeue -u pmirtaheri|squeue -u $USER|g' "$file"
    sed -i 's|squeue -u dbeaglehole|squeue -u $USER|g' "$file"

    # Replace hardcoded paths
    sed -i 's|/u/pmirtaheri/[^"'\'' ]*|/path/to/conda|g' "$file"
    sed -i 's|/u/dbeaglehole/[^"'\'' ]*|/path/to/project|g' "$file"
    sed -i 's|/work/hdd/bbjr/pmirtaheri/[^"'\'' ]*|/path/to/data|g' "$file"

    # Replace usernames in comments
    sed -i 's|pmirtaheri|USER|g' "$file"
    sed -i 's|dbeaglehole|USER|g' "$file"
done

# ============================================================================
# STEP 4: Anonymize markdown files
# ============================================================================
echo "Anonymizing markdown files..."

find "$OUTPUT_DIR" -name "*.md" -type f | while read -r file; do
    # Replace GitHub references
    sed -i 's|github.com/seyedparsa/|github.com/anonymous/|g' "$file"
    sed -i 's|github.com/dmbeaglehole/|github.com/anonymous/|g' "$file"
    sed -i 's|git+https://github.com/dmbeaglehole/|git+https://github.com/anonymous/|g' "$file"

    # Replace HuggingFace usernames
    sed -i 's|seyedparsa/|anonymous-user/|g' "$file"
done

# ============================================================================
# STEP 5: Clean Jupyter notebooks (remove outputs and metadata)
# ============================================================================
echo "Cleaning Jupyter notebooks..."

# Check if jupyter is available, otherwise use Python directly
if command -v jupyter &> /dev/null; then
    find "$OUTPUT_DIR" -name "*.ipynb" -type f | while read -r file; do
        jupyter nbconvert --clear-output --inplace "$file" 2>/dev/null || true
    done
else
    # Use Python to clean notebooks
    python3 << 'PYTHON_SCRIPT'
import os
import json
import sys

def clean_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Clear outputs and execution counts
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
            # Remove metadata that might contain identifying info
            cell_meta = cell.get('metadata', {})
            # Keep only essential metadata
            cell['metadata'] = {}

        # Clean notebook-level metadata
        nb_meta = nb.get('metadata', {})
        # Remove potentially identifying kernel info
        if 'kernelspec' in nb_meta:
            nb_meta['kernelspec'] = {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

        print(f"Cleaned: {filepath}")
    except Exception as e:
        print(f"Warning: Could not clean {filepath}: {e}", file=sys.stderr)

# Find and clean all notebooks
output_dir = os.environ.get('OUTPUT_DIR', 'anonymized_submission')
for root, dirs, files in os.walk(output_dir):
    for fname in files:
        if fname.endswith('.ipynb'):
            clean_notebook(os.path.join(root, fname))
PYTHON_SCRIPT
fi

# ============================================================================
# STEP 6: Create a clean README for submission
# ============================================================================
echo "Creating submission README..."

cat > "$OUTPUT_DIR/README.md" << 'EOF'
# Detecting Motivated Reasoning in Language Models

This repository contains code for detecting motivated reasoning in language models using supervised probes on residual-stream activations.

## Installation

```bash
pip install -r requirements.txt
pip install git+https://github.com/anonymous/xRFM.git  # Install xRFM
```

## Quick Start

```bash
# Generate responses
python main.py --generate --model qwen-3-8b --dataset arc-challenge --reason_first

# Generate with biased context
python main.py --generate --model qwen-3-8b --dataset arc-challenge --reason_first \
    --bias expert --hint_idx 0

# Evaluate responses
python main.py --evaluate --model qwen-3-8b --dataset arc-challenge

# Train probes
python main.py --train_probes --model qwen-3-8b --dataset arc-challenge \
    --probe has-switched --bias expert --n_ckpts 3 --ckpt rel

# Evaluate probes
python main.py --evaluate_probes --model qwen-3-8b --dataset arc-challenge \
    --probe has-switched --n_test_questions 200
```

## Project Structure

- `main.py` - Entry point for generation, evaluation, and probe training
- `thoughts/` - Core package for multiple-choice workflows
- `neural_controllers.py` - NeuralController class for steering and detection
- `control_toolkits.py` - Probe implementations (RFM, Linear, Logistic, PCA)
- `direction_utils.py` - Hidden state extraction and probe training utilities
- `analysis/` - Analysis and plotting scripts

## Supported Models

- Qwen3-8B
- Llama-3.1-8B-Instruct
- Gemma-3-4B

## Supported Datasets

- MMLU
- ARC-Challenge
- CommonsenseQA
- AQuA

## Hint/Bias Types

- `expert` (Sycophancy): "An expert indicates the answer is {X}"
- `self` (Consistency): Prefilled assistant response
- `metadata`: XML tags indicating correct choice
EOF

# ============================================================================
# STEP 7: Remove any remaining sensitive patterns
# ============================================================================
echo "Final cleanup - checking for remaining sensitive patterns..."

# Search for any remaining identifying patterns
PATTERNS="pmirtaheri|dbeaglehole|seyedparsa|ucsd\.edu|beaglehole"

FOUND_FILES=$(grep -rl -E "$PATTERNS" "$OUTPUT_DIR" 2>/dev/null || true)

if [ -n "$FOUND_FILES" ]; then
    echo ""
    echo "WARNING: Found potentially identifying patterns in the following files:"
    echo "$FOUND_FILES"
    echo ""
    echo "Please review these files manually:"
    grep -rn -E "$PATTERNS" "$OUTPUT_DIR" 2>/dev/null || true
else
    echo "No remaining identifying patterns found."
fi

# ============================================================================
# STEP 8: Create a zip archive
# ============================================================================
echo ""
echo "Creating zip archive..."
cd "$(dirname "$OUTPUT_DIR")"
zip -r "${OUTPUT_DIR}.zip" "$(basename "$OUTPUT_DIR")" -x "*.DS_Store" -x "*__MACOSX*"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Anonymization complete!"
echo "============================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Zip archive: ${OUTPUT_DIR}.zip"
echo ""
echo "Files excluded:"
echo "  - .git/ (git history)"
echo "  - .github/, .vscode/, .claude/ (config)"
echo "  - __pycache__/, .ipynb_checkpoints/ (cache)"
echo "  - core (crash dump)"
echo "  - *.err, *.out (job logs)"
echo "  - main.pdf (paper)"
echo "  - *.png, *.jpg (images)"
echo "  - simons*.yml (conda env)"
echo "  - outputs/, old_outputs/, figures/, data/, directions/"
echo "  - CLAUDE.md, AGENTS.md"
echo ""
echo "IMPORTANT: Please manually review the output before submission:"
echo "  1. Check notebooks for any remaining identifying content"
echo "  2. Review any warnings above"
echo "  3. Verify the code runs correctly"
echo ""
