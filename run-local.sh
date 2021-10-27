#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py \
--logdir='/data/eval/ActiveStereo_train' \
--config-file '/code/configs/local_train_gan.yaml' \
--summary-freq 1
