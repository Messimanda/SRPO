#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Student model path (same behavior as train.sh).
MODEL_PATH=${1:-/data/250010207/Vision-SR1/LLaMA-Factory-Cold-Start/saves/sft_models/Vision-SD-Cold-Start}

# External Teacher A model path.
TEACHER_A_MODEL_PATH=${2:-/data/share/Qwen2.5-14B-Instruct}

# Use 6 GPUs for training workers, leave 1 GPU budget for Teacher A, and keep 1 GPU as headroom.
TRAIN_N_GPUS=${TRAIN_N_GPUS:-6}
# Keep per-GPU effective batch close to the original 8-GPU setup for comparability.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-384}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-768}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-96}

# Keep the current training framework unchanged; only switch Teacher A source.
export VISION_SD_TEACHER_A_MODEL=${VISION_SD_TEACHER_A_MODEL:-${TEACHER_A_MODEL_PATH}}
export VISION_SD_TEACHER_A_NUM_VIEWS=${VISION_SD_TEACHER_A_NUM_VIEWS:-2}
export VISION_SD_TEACHER_A_FALLBACK_ON_ERROR=${VISION_SD_TEACHER_A_FALLBACK_ON_ERROR:-0}
export VISION_SD_TEACHER_A_TP=${VISION_SD_TEACHER_A_TP:-1}
# 14B teacher requires a substantially larger vLLM memory budget than 0.20.
export VISION_SD_TEACHER_A_GPU_UTIL=${VISION_SD_TEACHER_A_GPU_UTIL:-0.85}
# Keep context length moderate to avoid oversized KV cache at startup.
export VISION_SD_TEACHER_A_MAX_MODEL_LEN=${VISION_SD_TEACHER_A_MAX_MODEL_LEN:-4096}

python3 -m vision_sd.main \
    config=vision_sd/config.yaml \
    data.train_files=/data/250010207/Vision-SR1/datasets/Vision-SR1-47K@train \
    data.val_files=/data/250010207/Vision-SR1/datasets/mmstar@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.experiment_name=qwen2_5_vl_7b_credit_distill_grpo_external_teacher_a_6gpu_train \
    trainer.save_checkpoint_path=./saves/7b_grpo_credit_distill_external_teacher_a_6gpu_train \
    trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
