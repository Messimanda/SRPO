#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Student model path.
MODEL_PATH=${1:-/data/share/Qwen2.5-VL-3B-Instruct}

# Training/validation datasets (can be overridden by env vars).
TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/ViRL39K/39Krelease.parquet@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/MMK12@test}

# 8-GPU setup. Keep per-GPU load aligned with the 6-GPU recipe.
TRAIN_N_GPUS=${TRAIN_N_GPUS:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-384}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-768}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}

# Pure GRPO baseline: no extra credit weighting.
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
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.rollout.n=8 \
    worker.rollout.top_p=0.99 \
    algorithm.disable_kl=false \
    trainer.total_epochs=2 \
    trainer.experiment_name=qwen2_5_vl_3b_pure_grpo_baseline_8gpu \
    trainer.save_checkpoint_path=./saves/3b_pure_grpo_baseline_8gpu \
    trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
