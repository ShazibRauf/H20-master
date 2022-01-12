#!/bin/bash
bash /home/satti/H20-master-main/install.sh

export CUDA_VISIBLE_DEVICES=0
python3 HOPE.py > output.txt\
  --input_file /home/satti/H20-master-main/datasets/ho/ \
  --output_file /home/satti/H20-master-main/checkpoints/ho/model- \
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
  --pretrained_model /home/satti/H20-master-main/checkpoints/ho/model-85.pkl

