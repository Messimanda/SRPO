#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Student model path (3B).
MODEL_PATH=${1:-/data/share/Qwen2.5-VL-3B-Instruct}

# Training/validation datasets (can be overridden by env vars).
# Match train_simple_3s1g_3b.sh defaults.
TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/ViRL39K/39Krelease.parquet@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/MMK12@test}

# Resource/batch settings for 8-GPU training (same as train_simple_3s1g_3b.sh).
TRAIN_N_GPUS=${TRAIN_N_GPUS:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-384}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-768}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}

# RLSD perception credit; no external Teacher-A vLLM runner.
export VISION_SD_CREDIT_MODE=${VISION_SD_CREDIT_MODE:-simple_rlsd}
export VISION_SD_RUNNER_NUM_GPUS=${VISION_SD_RUNNER_NUM_GPUS:-0}

export VISION_SD_RLSD_WEAK_MODE=${VISION_SD_RLSD_WEAK_MODE:-mask_image}
export VISION_SD_RLSD_IMAGE_MASK_RATIO=${VISION_SD_RLSD_IMAGE_MASK_RATIO:-0.5}
export VISION_SD_RLSD_IMAGE_MASK_STYLE=${VISION_SD_RLSD_IMAGE_MASK_STYLE:-random_patch}
export VISION_SD_RLSD_PATCH_H=${VISION_SD_RLSD_PATCH_H:-14}
export VISION_SD_RLSD_PATCH_W=${VISION_SD_RLSD_PATCH_W:-14}
export VISION_SD_RLSD_W_CLIP_LOW=${VISION_SD_RLSD_W_CLIP_LOW:-0.8}
export VISION_SD_RLSD_W_CLIP_HIGH=${VISION_SD_RLSD_W_CLIP_HIGH:-1.2}
# Gate still uses lr_m: smaller k so negative lr_m does not slam k·lr_m (default was 6).
export VISION_SD_RLSD_GATE_K=${VISION_SD_RLSD_GATE_K:-2.0}
export VISION_SD_RLSD_NORMALIZE_PER_SEQ=${VISION_SD_RLSD_NORMALIZE_PER_SEQ:-1}
export VISION_SD_RLSD_LAMBDA_WARMUP_STEPS=${VISION_SD_RLSD_LAMBDA_WARMUP_STEPS:-0}
# Reward shaping before GRPO: R' = R + λ·max(0, Δ_mean). Default 0 = off; set env to re-enable.
export VISION_SD_RLSD_REWARD_DELTA_LAMBDA=${VISION_SD_RLSD_REWARD_DELTA_LAMBDA:-0}

# Teacher B on reasoning: "legacy" (z-norm+ReLU+gate+simple_mod) or "rlsvr" (RLVR paper: w=exp(sign(A)Δ), clip on reasoning only; still with-perception vs rollout logprob).
# export VISION_SD_SIMPLE_TEACHER_B_MOD=rlsvr

# Teacher-B gate: σ(β·(k·lr_m−δ)). Tuned (B): lower β (softer sigmoid), lower δ (shift right),
# so gate is not stuck ~0 when lr_m is slightly negative. Override via env if needed.
export VISION_SD_SIMPLE_GATE_DELTA=${VISION_SD_SIMPLE_GATE_DELTA:-0.0}
export VISION_SD_SIMPLE_GATE_BETA=${VISION_SD_SIMPLE_GATE_BETA:-2.0}
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
    trainer.experiment_name=qwen2_5_vl_3b_simple_rlsd_mask_8gpu \
    trainer.save_checkpoint_path=./saves/3b_simple_rlsd_mask_8gpu \
    trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
