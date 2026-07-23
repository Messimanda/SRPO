#!/bin/bash
#
# 7B: Perception = likelihood-ratio weights (w_t ∝ exp(Δ_t), per-seq mean norm, no tight w_clip).
#     Reasoning = rlsvr (same as train_simple_rlsd_rlsvr_7b.sh).
# Requires: VISION_SD_RLSD_PERCEPTION_CREDIT_ALLOC=proportional (see vision_sd/ray_trainer.py).

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=${1:-/data/share/Qwen2.5-VL-7B-Instruct}

TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/ViRL39K/39Krelease.parquet@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/MMK12@test}

NNODES=${NNODES:-1}
TRAIN_N_GPUS=${TRAIN_N_GPUS:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-384}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}

export VISION_SD_CREDIT_MODE=${VISION_SD_CREDIT_MODE:-simple_rlsd}
export VISION_SD_RUNNER_NUM_GPUS=${VISION_SD_RUNNER_NUM_GPUS:-0}

export VISION_SD_RLSD_WEAK_MODE=${VISION_SD_RLSD_WEAK_MODE:-mask_image}
export VISION_SD_RLSD_IMAGE_MASK_RATIO=${VISION_SD_RLSD_IMAGE_MASK_RATIO:-0.5}
export VISION_SD_RLSD_IMAGE_MASK_STYLE=${VISION_SD_RLSD_IMAGE_MASK_STYLE:-random_patch}
export VISION_SD_RLSD_PATCH_H=${VISION_SD_RLSD_PATCH_H:-14}
export VISION_SD_RLSD_PATCH_W=${VISION_SD_RLSD_PATCH_W:-14}
export VISION_SD_RLSD_W_CLIP_LOW=${VISION_SD_RLSD_W_CLIP_LOW:-0.8}
export VISION_SD_RLSD_W_CLIP_HIGH=${VISION_SD_RLSD_W_CLIP_HIGH:-1.2}
export VISION_SD_RLSD_GATE_K=${VISION_SD_RLSD_GATE_K:-2.0}
export VISION_SD_RLSD_NORMALIZE_PER_SEQ=${VISION_SD_RLSD_NORMALIZE_PER_SEQ:-1}
export VISION_SD_RLSD_LAMBDA_WARMUP_STEPS=${VISION_SD_RLSD_LAMBDA_WARMUP_STEPS:-0}
export VISION_SD_RLSD_REWARD_DELTA_LAMBDA=${VISION_SD_RLSD_REWARD_DELTA_LAMBDA:-0}

export VISION_SD_RLSD_PERCEPTION_CREDIT_ALLOC=${VISION_SD_RLSD_PERCEPTION_CREDIT_ALLOC:-proportional}
export VISION_SD_SIMPLE_PERCEPTION_FLOOR=${VISION_SD_SIMPLE_PERCEPTION_FLOOR:-0.0}

export VISION_SD_SIMPLE_TEACHER_B_MOD=${VISION_SD_SIMPLE_TEACHER_B_MOD:-rlsvr}
export VISION_SD_TEACHER_B_PAPER_W_EPS=${VISION_SD_TEACHER_B_PAPER_W_EPS:-0.2}
export VISION_SD_TEACHER_B_PAPER_DELTA_CLIP=${VISION_SD_TEACHER_B_PAPER_DELTA_CLIP:-20.0}

export VISION_SD_SIMPLE_GATE_DELTA=${VISION_SD_SIMPLE_GATE_DELTA:-0.0}
export VISION_SD_SIMPLE_GATE_BETA=${VISION_SD_SIMPLE_GATE_BETA:-2.0}
export VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE=${VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE:-0.6}
export VISION_SD_SIMPLE_MOD_LAMBDA=${VISION_SD_SIMPLE_MOD_LAMBDA:-0.5}
export VISION_SD_SIMPLE_MOD_CLIP=${VISION_SD_SIMPLE_MOD_CLIP:-0.4}
export VISION_SD_SIMPLE_MOD_MULT_MIN=${VISION_SD_SIMPLE_MOD_MULT_MIN:-0.3}
export VISION_SD_SIMPLE_MOD_MULT_MAX=${VISION_SD_SIMPLE_MOD_MULT_MAX:-1.7}

python3 -m vision_sd.main \
    config=vision_sd/config_changed_7b.yaml \
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
    trainer.experiment_name=qwen2_5_vl_7b_simple_rlsd_proportional_rlsvr_8gpu \
    trainer.save_checkpoint_path=/data/250010207/Vision-SR1/saves/7b_simple_rlsd_proportional_rlsvr_8gpu \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
