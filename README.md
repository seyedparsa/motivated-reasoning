## Minimum working requirements


## Usage

### Generation

Answer to unbiased context with CoT:
```
python main.py \
--generate \
--model {model} \
--dataset {dataset} \
--reason_first 
```

Remove ```--reason_first``` to answer without CoT.
Add the following to answer to biased context:
```
--bias {bias_type} \ # in ['expert', 'self', 'metadata']
--hint_idx {hint_index} \ # 0 <= hint_idx < num_choices
```

Optional:
```
--split {split} \
--n_gen {num_data}
--bs_gen {batch_size}
```


### Evaluation
```
python main.py \
--evaluate \
--model {model} \
--dataset {dataset} \
```

Optional:
```
--split {split}
```

### Probing
```
python thoughts_eval.py \                                                                                                                                   
--model {model} \
--dataset {dataset} \
--split {split} \
--probe {task} \
--bias {bias_type} \
--bs_probe {batch_size} \
--n_ckpt {num_checkpoints}
```

Example:
```
python main.py \
--model qwen-3-8b \
--dataset arc-challenge \
--probe has-switched \
--bias self \
--bs_probe 8 \
--n_ckpt 2
```


### Steering


## Citation
