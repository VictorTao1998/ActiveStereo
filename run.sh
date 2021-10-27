#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/StereoGAN/train.py \
--logdir='/jianyu-fast-vol/eval/StereoGAN_train_1' \
--config-file '/jianyu-fast-vol/StereoGAN/configs/remote_train_gan.yaml'
--summary-freq 50
