#!/bin/bash

deepspeed --master_port 12345 tools/train_robocasa.py \
    --deepspeed \
    --deepspeed_config configs/ds_bf16_z3_config.json \
    --name geopredict_robocasa \
    --pretrain ./ckpts/pi0_base.pth \
    --num_train_steps 40000 \
    --optimizer AdamW \
    --warmup_steps 1000 \
    --decay_steps 30000 \
    --log_interval 100 \
    --save_interval 5000 \
    --batch_size 4
