#!/bin/bash
bash /netscratch/satti/H20-master-main-HandAndObjMesh-NTAM-TSDF/install.sh

export CUDA_VISIBLE_DEVICES=0
python3 HOPE.py > output.txt\
  --input_file /netscratch/satti/H20-master-main-HandAndObjMesh-NTAM-TSDF/datasets/ho/ \
  --output_file /netscratch/satti/H20-master-main-HandAndObjMesh-NTAM-TSDF/checkpoints/ho/model- \
  --train \
  --val \
  --stage 0 \
  --batch_size 32 \
  --model_def HopeNet \
  --gpu \
  --gpu_number 0 \
  --learning_rate 0.0001 \
  --lr_step 1000 \
  --lr_step_gamma 0.9 \
  --log_batch 100 \
  --val_epoch 1 \
  --snapshot_epoch 1 \
  --num_iterations 700 \
  --pretrained_model /netscratch/satti/H20-master-main-HandAndObjMesh-NTAM-TSDF/checkpoints/ho/model-33.pkl

