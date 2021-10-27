#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/ActiveStereo/main.py \
--logdir='/jianyu-fast-vol/eval/ActiveStereo_train' \
--config-file '/jianyu-fast-vol/ActiveStereo/configs/remote_train_gan.yaml' \
--summary-freq 50
