
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