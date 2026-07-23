#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Student model path.
MODEL_PATH=${1:-/data/share/Qwen2.5-VL-3B-Instruct}

# Training/validation datasets (can be overridden by env vars).
TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/ViRL39K/39Krelease.parquet@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/MMK12@test}

# 8-GPU setup.
TRAIN_N_GPUS=${TRAIN_N_GPUS:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-384}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}

# Keep advantage path pure (no extra credit weighting), then apply DAPO-style update knobs.
export VISION_SD_CREDIT_MODE=${VISION_SD_CREDIT_MODE:-pure_grpo}

python3 -m vision_sd.main \
    config=vision_sd/config.yaml \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.image_dir=/data/250010207/Vision-SR1/datasets/ViRL39K \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_response_length=2048 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.n=8 \
    worker.rollout.top_p=0.99 \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.online_filtering=true \
    trainer.total_epochs=2 \
    trainer.experiment_name=qwen2_5_vl_3b_dapo_baseline_8gpu \
    trainer.save_checkpoint_path=./saves/3b_dapo_baseline_8gpu \
    trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
