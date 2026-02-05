# Neural Controllers Copilot Instructions

## Project Overview
This is a research library for implementing neural controllers with decoder-only LLMs, enabling concept steering and detection. The core approach uses layer-wise predictors (primarily Recursive Feature Machines/RFMs) to influence model behavior during generation or detect specific concepts in text.

## Core Architecture

### Main Components
- **`NeuralController`** (`neural_controllers.py`): Central orchestrator that coordinates steering/detection
- **Control Toolkits** (`control_toolkits.py`): Pluggable algorithms (`rfm`, `linear`, `logistic`, `pca`, `mean_difference`)
- **Direction Utils** (`direction_utils.py`): Core mathematical operations for extracting hidden states and training predictors
- **Generation Utils** (`generation_utils.py`): Hook-based text generation with real-time steering

### Data Flow Pattern
1. **Hidden State Extraction**: Text → Model → Layer-wise hidden states (dict with negative layer indices: `-1`, `-2`, etc.)
2. **Direction Training**: Hidden states + labels → Layer-specific control vectors/detectors
3. **Steering/Detection**: Apply control vectors during generation OR evaluate text with trained detectors

## Critical Development Patterns

### Model Initialization
```python
# Standard pattern across notebooks
controller = NeuralController(
    language_model,
    tokenizer,
    control_method='rfm',  # or 'linear', 'logistic', etc.
    n_components=5,
    rfm_iters=8,
    batch_size=2
)
```

### Layer Indexing Convention
- **Negative indexing**: `-1` = final layer, `-2` = second-to-last, etc.
- **Hidden layers list**: `controller.hidden_layers` = `[-1, -2, ..., -num_layers]`
- **Control targeting**: `layers_to_control=list(range(-1, -31, -1))` for deep steering

### Data Structure Patterns
- **Hidden states**: `Dict[int, torch.Tensor]` where keys are negative layer indices
- **Labels**: Always converted to `torch.Tensor.reshape(-1,1)` for consistency
- **Metrics**: Nested dicts `{layer_idx: {'auc': float, 'acc': float, ...}, 'aggregated': {...}}`

## Key Workflows

### Training Directions
```python
# Compute steering directions
controller.compute_directions(train_data, train_labels, val_data, val_labels)

# Evaluate detection performance
val_metrics, test_metrics, _ = controller.evaluate_directions(
    train_data, train_labels, val_data, val_labels, test_data, test_labels
)
```

### Controlled Generation
```python
# Generate with steering
output = controller.generate(
    prompt,
    layers_to_control=list(range(-1, -11, -1)),  # Control last 10 layers
    control_coef=0.5,  # Steering strength
    max_new_tokens=150
)
```

## File Organization Conventions

### Notebooks (`notebooks/`)
- Each concept has dedicated notebook (e.g., `shakespeare.ipynb`, `poetry.ipynb`)
- Standard pattern: Model setup → Data loading → Direction training → Generation examples
- Use `%load_ext autoreload` and sys.path modification for development

### Data Structure (`data/`)
- Concept-specific subdirectories (e.g., `data/languages/` for translation)
- Expected by utility functions like `shakespeare_dataset()` in `utils.py`

### Directions (`directions/`)
- Saved models: `{method}_{concept}_{model_name}_{detector?}.pkl`
- Example: `rfm_poetry_llama_3_8b_it.pkl`

### Execution Scripts
- SLURM-based: Use `submit_*.sh` for cluster execution
- Local testing: Use `run_*.sh` scripts
- Standard args: `--model`, `--dataset`, `--bias`, `--n_gen`, `--bs_gen`

## Development Guidelines

### Adding New Control Methods
1. Implement toolkit class inheriting from `Toolkit` in `control_toolkits.py`
2. Add to `TOOLKITS` dict in `neural_controllers.py`
3. Ensure `_compute_directions()` returns: `(directions, signs, detector_coefs, accuracies)`

### Working with Hidden States
- Always check if data is pre-computed: `isinstance(data, dict)` indicates hidden states
- Use `direction_utils.get_hidden_states()` for extraction with batching
- Memory management: Consider `forward_batch_size` parameter for large datasets

### Hook-Based Generation
- Hooks modify layer outputs during forward pass: `output[0] += control_coef * control_vec`
- Always clear hooks after generation: `generation_utils.clear_hooks(hooks)`
- Control vectors shape: `(1, 1, hidden_dim)` for broadcasting

### External Dependencies
- **xRFM**: Install from `git+https://github.com/dmbeaglehole/xRFM.git`
- **HuggingFace Models**: Llama-3.1-8B-Instruct, Gemma-2-9B-it commonly used
- **Device handling**: Models typically loaded with `device_map="cuda"`

## Testing & Validation
- Use `controller.describe()` for debugging setup
- Layer-wise accuracy in `evaluate_directions()` helps identify optimal control layers
- Steering strength (`control_coef`) typically ranges 0.1-1.0
- Monitor generation quality vs. control strength trade-offs

## Advanced Evaluation Features

### GPT-4o Articulation Judgment
- `articulates_influence()` in `thoughts/multiple_choice.py` uses GPT API to evaluate if model responses show explicit influence from hints
- Requires `OPENAI_API_KEY` environment variable
- Implements structured evaluation similar to academic paper standards
- Falls back to keyword-based detection if API fails