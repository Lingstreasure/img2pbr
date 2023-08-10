#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python src/eval.py \
    experiment=pbr.yaml \
    ckpt_path=/media/d5/7D1922F98D178B12/hz/Code/img2pbr/logs/pbr_reconstruction/train/runs/2023-08-08_22-38-08/checkpoints/epoch_099.ckpt \
    task_name=pbr_reconstruction/eval
