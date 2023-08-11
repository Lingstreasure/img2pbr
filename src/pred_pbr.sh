#!/bin/bash
python src/predict_pbr.py \
    experiment=pbr \
    ckpt_path=/media/d5/7D1922F98D178B12/hz/Code/img2pbr/logs/pbr_reconstruction/train/runs/2023-08-11_09-45-40/checkpoints/epoch_075.ckpt \
    task_name=pbr_reconstruction/predict \
    trainer.devices=1
