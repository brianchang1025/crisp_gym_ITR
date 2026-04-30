#!/bin/bash

# replace with your dataset repo e.g. cbrian/pi05_test_dataset
DATASET_REPO=username/dataset_repo_name
# replace with your output directory e.g. outputs/pi05_test_training
# NOTE: Keep "/workspace/outputs/" prefix, only change the folder name after it
OUTPUT_DIR=/workspaceoutputs/output_folder_name
# replace with your job name e.g. pi05_test_job
JOB_NAME=jobname
# replace with your policy repo e.g. cbrian/pi05_test_trained_policy
POLICY_REPO=username/policy_repo_name

STEPS=2000

lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=groot \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO" \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.tune_diffusion_model=false \
    --steps=$STEPS \
    --policy.device=cuda:0 \
    --batch_size=32 \
    --eval_freq 500 \
    --eval.n_episodes 50 \
    --save_checkpoint true \
    --save_freq 500 \
    --log_freq=100 \

# Use pixi run -e jazzy-pi05 python -m lerobot.scripts.lerobot_train --help to see all available options and their descriptions.