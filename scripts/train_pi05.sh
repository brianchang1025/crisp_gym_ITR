#!/bin/bash

# replace with your dataset repo e.g. cbrian/pi05_test_dataset
DATASET_REPO=cbrian/dataset_env_SST_SP9_WC1_TC1_task_pickplaceblueblock_numepi_10_ctrl_cartesian
# replace with your output directory e.g. outputs/pi05_test_training
# NOTE: Keep "outputs/" prefix, only change the folder name after it
OUTPUT_DIR=outputs/output_folder_name
# replace with your job name e.g. pi05_test_training
JOB_NAME=jobname
# replace with your policy repo e.g. cbrian/pi05_test_trained_policy
POLICY_REPO=username/policy_name

pixi run -e jazzy-pi05 python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=pi05 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO" \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=true \
    --steps=100 \
    --policy.device=cuda:0 \
    --batch_size=2 \
    --eval_freq 100 \
    --eval.n_episodes 50 \
    --save_checkpoint true \
    --save_freq 100 \

# Use pixi run -e jazzy-pi05 python -m lerobot.scripts.lerobot_train --help to see all available options and their descriptions.