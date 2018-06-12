## Major Changes

### 1. EMA variables

This modification given a trained model continues to train a model while keeping shadow variables of weights. To train with EMA add these flags:
`--ema_source "path_to/source_model/inference_model"` and `--ema_halflife 2000`. Adjusting the flag parameters accordingly.

`ema_halflife` parameter has to be optimized. For starters I would recommend using vaue of `1500 X NGPUs X BathcSize / 512`
 and train it for 2 epochs at reduced (1/5) Learning Rate. 

To activate the shadow variables for evaluation use flag `--use_EMA` with script `eval.py`.

 
### 2. Graph ensemble script 

`graph_ensemble.py` takes in 2 or more trained models and combines them into a single graph. Sample usage:
```
python graph_ensemble.py
  --models "/path/to/model1/inference_model" "/path/to/model2/inference_model\"
  --save_folder "/path/to/joined_model"
  --weights 0.7 0.3
```
