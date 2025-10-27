## Minimum working requirements


## Usage

Answer to unbiased context with CoT:
```
python main.py \
--generate \
--model {model} \
--dataset {dataset} \
--split {split} \
--n_gen {num_data}
--bs_gen {batch_size} \
--reason_first 
```

Remove ```--reason_first``` to answer without CoT.
Add the following to answer to biased context:
```
--bias {bias_type} \ # in ['expert', 'self', 'metadata']
--hint_idx {hint_index} \ # 0 <= hint_idx < num_choices
```


## Citation
