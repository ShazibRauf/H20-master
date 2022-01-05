#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
python3 HOPE.py > output.txt\
  --input_file /home5/satti/HOPE-master9-Zeros-cam4/datasets/ho/ \
  --output_file /home5/satti/HOPE-master9-Zeros-cam4/checkpoints/ho/model- \
  --train \
  --val \
  --stage 2 \
  --batch_size 32 \
  --model_def HopeNet \
  --gpu \
  --gpu_number 0 \
  --learning_rate 0.001 \
  --lr_step 1000 \
  --lr_step_gamma 0.9 \
  --log_batch 100 \
  --val_epoch 1 \
  --snapshot_epoch 1 \
  --num_iterations 120 \
  --pretrained_model ./checkpoints/ho/model-101.pkl

