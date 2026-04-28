#!/bin/bash

pixi run -e jazzy-pi05 python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=username/repo_name \
    --policy.type=pi05 \
    --output_dir=outputs/output_folder_name \
    --job_name=jobname \
    --policy.repo_id=username/policy_name \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=true \
    --steps=15000 \
    --policy.device=cuda:0 \
    --batch_size=32 \
    --eval_freq 2500 \
    --eval.n_episodes 50 \
    --save_checkpoint true \
    --save_freq 10 \