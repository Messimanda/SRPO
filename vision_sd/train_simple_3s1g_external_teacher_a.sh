#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Student model path.
MODEL_PATH=${1:-/data/share/Qwen2.5-VL-7B-Instruct}

# External Teacher A model path (independent from student).
TEACHER_A_MODEL_PATH=${2:-/data/share/Qwen2.5-14B-Instruct}

# Training/validation datasets (can be overridden by env vars).
TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/ViRL39K/39Krelease.parquet@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/MMK12-8K/train/data.parquet@train}

# Default: 6 GPUs for training workers + 1 GPU for Runner(Teacher A vLLM).
# Keep per-GPU load aligned with the 8-GPU recipe by scaling batch sizes by 6/8.
TRAIN_N_GPUS=${TRAIN_N_GPUS:-6}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-288}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-576}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-96}

# Use simple 3-signal + 1-gate mode.
export VISION_SD_CREDIT_MODE=${VISION_SD_CREDIT_MODE:-simple_3s1g}

# This script explicitly uses a standalone external Teacher A engine.
export VISION_SD_TEACHER_A_BACKEND=${VISION_SD_TEACHER_A_BACKEND:-vllm}
export VISION_SD_TEACHER_A_MODEL=${VISION_SD_TEACHER_A_MODEL:-${TEACHER_A_MODEL_PATH}}
export VISION_SD_TEACHER_A_NUM_VIEWS=${VISION_SD_TEACHER_A_NUM_VIEWS:-1}
export VISION_SD_TEACHER_A_FALLBACK_ON_ERROR=${VISION_SD_TEACHER_A_FALLBACK_ON_ERROR:-0}
export VISION_SD_TEACHER_A_TP=${VISION_SD_TEACHER_A_TP:-1}
export VISION_SD_TEACHER_A_GPU_UTIL=${VISION_SD_TEACHER_A_GPU_UTIL:-0.85}
export VISION_SD_TEACHER_A_MAX_MODEL_LEN=${VISION_SD_TEACHER_A_MAX_MODEL_LEN:-4096}

# Allocate 1 GPU to Runner so Teacher A vLLM can start.
export VISION_SD_RUNNER_NUM_GPUS=${VISION_SD_RUNNER_NUM_GPUS:-1}

# Three signals + one gate (default conservative values).
export VISION_SD_SIMPLE_EXT_REL_COEF=${VISION_SD_SIMPLE_EXT_REL_COEF:-0.5}
export VISION_SD_SIMPLE_EXT_FACT_COEF=${VISION_SD_SIMPLE_EXT_FACT_COEF:-0.5}
export VISION_SD_SIMPLE_EXT_LEAK_COEF=${VISION_SD_SIMPLE_EXT_LEAK_COEF:-0.35}
export VISION_SD_SIMPLE_GATE_LAMBDA_EXT=${VISION_SD_SIMPLE_GATE_LAMBDA_EXT:-0.6}
export VISION_SD_SIMPLE_GATE_LAMBDA_SOLVE=${VISION_SD_SIMPLE_GATE_LAMBDA_SOLVE:-0.4}
export VISION_SD_SIMPLE_GATE_DELTA=${VISION_SD_SIMPLE_GATE_DELTA:-0.35}
export VISION_SD_SIMPLE_GATE_BETA=${VISION_SD_SIMPLE_GATE_BETA:-8.0}

# Utility approximation: top-k mask or random_drop.
export VISION_SD_SIMPLE_UTILITY_MODE=${VISION_SD_SIMPLE_UTILITY_MODE:-topk_mask}
export VISION_SD_SIMPLE_UTILITY_TOPK=${VISION_SD_SIMPLE_UTILITY_TOPK:-3}
export VISION_SD_SIMPLE_UTILITY_DROP_RATIO=${VISION_SD_SIMPLE_UTILITY_DROP_RATIO:-0.3}

# Stable credit modulation.
export VISION_SD_SIMPLE_PERCEPTION_FLOOR=${VISION_SD_SIMPLE_PERCEPTION_FLOOR:-0.35}
export VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE=${VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE:-0.6}
export VISION_SD_SIMPLE_MOD_LAMBDA=${VISION_SD_SIMPLE_MOD_LAMBDA:-0.5}
export VISION_SD_SIMPLE_MOD_CLIP=${VISION_SD_SIMPLE_MOD_CLIP:-0.4}
export VISION_SD_SIMPLE_MOD_MULT_MIN=${VISION_SD_SIMPLE_MOD_MULT_MIN:-0.3}
export VISION_SD_SIMPLE_MOD_MULT_MAX=${VISION_SD_SIMPLE_MOD_MULT_MAX:-1.7}

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
    worker.rollout.n=8 \
    worker.rollout.top_p=0.99 \
    trainer.total_epochs=2 \
    trainer.experiment_name=qwen2_5_vl_7b_simple_3s1g_external_teacher_a_6gpu \
    trainer.save_checkpoint_path=./saves/7b_simple_3s1g_external_teacher_a_6gpu \
    trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
