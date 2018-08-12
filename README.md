# Next top GB model solution to The 2nd YouTube-8M Video Understanding Challenge

This repository contains all code used by first placed team "Next top GB model" ([David](https://www.kaggle.com/tivfrvqhs5) and [Miha skalic](https://www.kaggle.com/mihaskalic)) in the Kaggle's competition [The 2nd YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m-2018/).
 
Code is released under Apache License Version 2.0.


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

### 3. Quantization 

there are the 3 ways to reduce model size:
1. converting variables to float16 (approx. 1/2 the size)
2. 8bit uniform min-max quantization (approx. 1/4 the size)
3. 8bit quantization with where quants are determined by k-means clustering (approx. 1/4 the size)

Sample usage
```
python quantize.py \
  --transform_type quant_uniform \
  --model ../path/to/modelX/inference_model \
  --min_elements 17000 \
  --savefile ../path/to/modelX/inference_model_uniformquant
```
by default only vaiables with more than 17000 parameters will be quantized. Use parameter `--min_elements` to adjust 
the threshold. 


### 4. fake_utls
In this subfolder you will find commands to clear redundant test samples. Follow the readme in the folder.


### 5. Changes to eval.py

If `build_only` flag is set the script will compile the model (you still have to provide the parameters) without the data providers. 
This is useful if you want to run models on other machines. In this case evaluation loop will not run. 


The default eval.py will not work on custom quantized and or ensembled graphs. For this use `eval_custom.py`:

```
python eval_custom.py \
  --model_file="../trained_models/xxx/inference_model"  \
  --batch_size 150 \
  --eval_data_pattern="../data/frame/validate/*A.tfrecord"
```

### 6. Model Distillation

`prepare_distill_dataset.py` and `train_distill.py` are scripts used in distillation training. Former is used to generated distilled dataset, 
while the later one used is used to train new models on that dataset. Flags are documented within the scripts. 
 