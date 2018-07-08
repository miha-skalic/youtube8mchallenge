
# Commands to train models


**NOTE: do not forget to copy aditional validation data (except suggested 800 files) to training dir**

-  Export Devices: export CUDA_VISIBLE_DEVICES=0,1
-  Set path to save folder and record folder

Set path to save folder and record folder
```
SAVEPATH="../trained_models"
RECORDPAT="../data/frame/train"
```

Run following two commands:
```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=Lstmbidirect \
  --video_level_classifier_model="LogisticModel" \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=256 \
  --train_dir="$SAVEPATH//LSTM_bidirectv0_logistic" \
  --base_learning_rate=0.0002 \
  --lstm_cells=1024 \
  --num_epochs=7 \
  --num_gpu 2 \
  --num_readers 12 \
  --start_new_model
```

```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=Lstmbidirect \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=256 \
  --train_dir="$SAVEPATH/LSTM_bidirectv0_MoE" \
  --base_learning_rate=0.0002 \
  --lstm_cells=1024 \
  --num_epochs=7 \
  --num_gpu 2 \
  --num_readers 12 \
  --start_new_model
```


## NetVLAD gatting 

Set path to save folder and record folder

```
SAVEPATH="../trained_models"
RECORDPAT="../data/frame/train"
```

```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir="$SAVEPATH/gatednetvladLF_v0" \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=80 \
  --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --num_gpu 2 \
  --num_epochs=8
```

## BN dual-path GRU 
```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=GRUbidirect_branchedBN \
  --video_level_classifier_model="LogisticModel" \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=256 \
  --train_dir="$SAVEPATH/GRU_BN_dualpath" \
  --base_learning_rate=0.0002 \
  --num_epochs=7 \
  --num_gpu 2 \
  --num_readers 12
```

## DBoF model
```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=DbofModel \
  --video_level_classifier_model="LogisticModel" \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=512 \
  --train_dir="$SAVEPATH/DBoFv0" \
  --num_epochs=7 \
  --num_gpu 2 \
  --iterations=40 \
  --num_readers 12
```

# RV netVLAD

```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir="$SAVEPATH/gatedlightvladLF_v0" \
  --frame_features=True --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=80 --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --lightvlad=True \
  --num_gpu 2 \
  --num_epochs=7 \
```


# 
```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=Lstmbidirect \
  --video_level_classifier_model="LogisticModel" \
  --frame_features \
  --feature_names="rgb, audio" \
  --feature_sizes="1024, 128" \
  --batch_size=256 \
  --train_dir="$SAVEPATH//LSTM_bidirectv0_frameShuffle" \
  --base_learning_rate=0.0002 \
  --lstm_cells=1024 \
  --num_epochs=7 \
  --num_gpu 2 \
  --num_readers 12 \
  --frame_shuffle
  --start_new_model
```

```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=GatedDbofModelLF \
  --train_dir="$SAVEPATH/gatedsoftdbof" \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=80 \
  --base_learning_rate=0.0002 \
  --dbof_cluster_size=4096 \
  --dbof_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=300 \
  --dbof_relu=False \
  --num_gpu 2 \
  --num_epochs=7
```

```
python train.py \
  --train_data_pattern="$RECORDPAT/*.tfrecord" \
  --model=SoftDbofModelLF \
  --train_dir=softdboflf8000 \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=80 \
  --base_learning_rate=0.0002 \
  --dbof_cluster_size=8000 \
  --dbof_hidden_size=1024 \
  --iterations=300 \
  --dbof_relu=False \
  --max_step=800000
```
