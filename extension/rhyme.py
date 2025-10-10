import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController
import utils

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

language_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda"
)

# use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
use_fast_tokenizer = False
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
model_name='llama_3_8b_it'
assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'

# data_dir = "neural_controllers/data/poetry"

# dataset = utils.poetry_dataset(data_dir=data_dir, tokenizer=tokenizer, assistant_tag=assistant_tag)

dataset = utils.rhyme_dataset(data_path="rhyme_dataset_muse.csv", tokenizer=tokenizer)
# print(dataset)

controller = NeuralController(
    language_model,
    tokenizer,
    rfm_iters=8,
    batch_size=2,
    control_method='logistic'
)

controller.compute_directions(dataset['train']['inputs'], dataset['train']['labels'])