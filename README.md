# Next top GB model solution to The 2nd YouTube-8M Video Understanding Challenge

This repository contains all code used by first placed team "Next top GB model" ([David](https://www.kaggle.com/tivfrvqhs5) and [Miha skalic](https://www.kaggle.com/mihaskalic)) in the Kaggle's competition [The 2nd YouTube-8M Video Understanding Challenge](https://www.kaggle.com/c/youtube8m-2018/).
 
The repository is a fork of [google's repository](https://github.com/google/youtube-8m) and borrows from [Wang et al](https://github.com/wangheda/youtube-8m), [Miech et al](https://github.com/antoine77340/Youtube-8M-WILLOW) and [Skalic et al](https://github.com/mpekalski/Y8M). Code is released under Apache License Version 2.0.

This readme walks through a specific example to reproduce training, eval, distillation, quantization, graph combination for a single model type.


## Citation

```
@inproceedings{skalic2018building,
  title={Building A Size Constrained Predictive Models for Video Classification},
  author={Skalic, Miha and Austin, David},
  booktitle={European Conference on Computer Vision},
  pages={297--305},
  year={2018},
  organization={Springer}
}
```


## Background

All models herein were trained in single GPU mode and the instructions that follow will reproduce this step. The
overall flow for training each model is as follows:
1. Train model
2. Evaluate model
3. (Optional) Perform EMA- Exponentially weighted Moving Average of weights
4. Quantize model
5. Perform inference on quantized model
6. (Optional) If running distillation, create distillation dataset, then repeat from step 1.
7. Combine multiple graphs into single graph


This readme will walk through all commands to train both stand-alone and a distillation model.  For model details for other models
see all_models.txt.

## Requirements

All code was run using Python 2.7 and Tensorflow 1.8.0.  All models were trained on GPU's.  The requirements.txt file
contains a list of all libraries installed in the environment used for training and testing.  While all libraries are not
required, the having the full list should ensure complete compatibility with all code.



## Training Models

Make sure to set your local paths correctly for the train and save paths:

```
export CUDA_VISIBLE_DEVICES=0
SAVEPATH="../trained_models"
RECORDPAT="../data/frame/train"


python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir="$SAVEPATH//NetVLAD" \
  --frame_features=True --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=160 --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --lightvlad=False \
  --num_gpu 1 \
  --num_epochs=10 \
```

# Eval model

Once training is complete, eval is performed as follows:

```
RECORDPATVAL="../data/frame/train"

python eval.py \
  --eval_data_pattern="$RECORDPATVAL/*.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir="$SAVEPATH//NetVLAD" \
  --frame_features=True --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=160 --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --lightvlad=False \
  --num_gpu 1 \
  --num_epochs=10 \
  --run_once \
  --build_only \
  --sample_all
```

# Perform EMA

```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir="$SAVEPATH//NetVLAD_ema" \
  --video_level_classifier_model="LogisticModel" \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=160 \
  --base_learning_rate=0.00008 \
  --lstm_cells=1024 \
  --num_epochs=2 \
  --num_gpu 1 \
  --num_readers 8 \
  --loss_lambda 0.5 \
  --ema_halflife 2000 \
  --ema_source "$SAVEPATH//NetVLAD/inference_model"

python eval.py \
  --eval_data_pattern="$RECORDPATVAL/*.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir="$SAVEPATH//NetVLAD_ema" \
  --video_level_classifier_model="LogisticModel" \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=160 \
  --base_learning_rate=0.00008 \
  --lstm_cells=1024 \
  --num_epochs=2 \
  --num_gpu 1 \
  --num_readers 8 \
  --build_only \
  --run_once \
  --sample_all

```

# Quantize Model and copy model_flags.json

Change savefile to specific save path

```

python quantize.py \
  --transform_type quant_uniform \
  --model "$SAVEPATH//NetVLAD_ema/inference_model" \
  --savefile ../trained_models/quants/your_model/inference_model

cp $SAVEPATH//NetVLAD_ema/model_flags.json ../trained_models/quants/your_model

```
# Combine multiple graphs into single graph

graph_ensemble.py takes in 2 or more trained models and combines them into a single graph. Sample usage:


```
python graph_ensemble.py \
--models ../trained_models/quants/74/inference_model \
        ../trained_models/model_1/inference_model \
        ../trained_models/model_2/inference_model \
        ../trained_models/model_3/inference_model \
--weights 0.3333 0.3333 0.3334  \
--save_folder ../train_models/your_combined_output_graph
```


# Perform Inference

```

RECORDPATTEST="../data/frame/test"

python inference_gpu.py \
  --train_dir "../trained_models/quants/your_model"  \
  --output_file="./output.csv" \
  --input_data_pattern="$RECORDPATtest/*.tfrecord" \
  --batch_size 200 \
  --sample_all
```

# Create Distillation Set and Train on it.

WARNING: Large dataset creation!  Creating a new Distillation set will consume ~1.4TB of data so you'll need
to have the storage space available.

```
python prepare_distill_dataset.py   --batch_size 128   --file_size 512   --input_data_pattern "$RECORDPATVAL/*.tfrecord"   --output_dir "output_folder/train_distill/"   --model_file "../train_models/your_ensemble_model/inference_model"

```

Training on a distillation dataset can be done using `train_distill.py` script. Use same flags as in `train.py`.

# Model configurations

File `model_configs.xlsx` contains the arhitectures of models used in the work.

# Trained Model 

Trained model as `.tar.gz` can be downloaded from [here](https://drive.google.com/open?id=1hrHOWc_3xFk1WofTnimq8icjzJ-k9pnh).
See `inference.py` for sample usage of the model. Folder `feature_extractor` contains information on 
preprocessing custom videos. 
