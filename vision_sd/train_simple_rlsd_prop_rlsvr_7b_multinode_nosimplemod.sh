#!/bin/bash
#
# 7B: like train_simple_rlsd_prop_rlsvr_rollout_qi_7b_multinode_dapo_entpen0p03.sh
# but disables simple_mod:
#   VISION_SD_SIMPLE_MOD_BYPASS=1  -> modulation = raw_w (no simple_mod reshape).
# Keep rollout-vs-QI reasoning delta and entropy penalty aligned with baseline:
#   VISION_SD_TEACHER_B_REASONING_DELTA_MODE=rollout_vs_qi
#   worker.actor.entropy_penalty_coef=0.03
# Difference from baseline: simple_mod is disabled only.
#
# Launch:
#   bash vision_sd/train_simple_rlsd_prop_rlsvr_7b_multinode_nokl_nosimplemod.sh

set -euo pipefail
set -x

MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-23456}
NNODES=${NNODES:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-2}}}
NODE_RANK=${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-0}}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}

MODEL_PATH=${1:-/data/share/Qwen2.5-VL-7B-Instruct}

TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/ViRL39K/39Krelease.parquet@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/MMK12@test}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-384}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONUNBUFFERED=1
export VISION_SD_CREDIT_MODE=${VISION_SD_CREDIT_MODE:-simple_rlsd}
export VISION_SD_RUNNER_NUM_GPUS=${VISION_SD_RUNNER_NUM_GPUS:-0}
export VISION_SD_TEACHER_B_REASONING_DELTA_MODE=${VISION_SD_TEACHER_B_REASONING_DELTA_MODE:-rollout_vs_qi}
export VISION_SD_QI_SCORING_FORMAT=${VISION_SD_QI_SCORING_FORMAT:-"${REPO_ROOT}/vision_sd/format_prompt/qi_only_reasoning_answer.jinja"}

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
export VISION_SD_TEACHER_B_PAPER_DELTA_CLIP=${VISION_SD_TEACHER_B_PAPER_DELTA_CLIP:-3.0}

export VISION_SD_SIMPLE_GATE_DELTA=${VISION_SD_SIMPLE_GATE_DELTA:-0.0}
export VISION_SD_SIMPLE_GATE_BETA=${VISION_SD_SIMPLE_GATE_BETA:-2.0}
export VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE=${VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE:-0.6}
export VISION_SD_SIMPLE_MOD_BYPASS=${VISION_SD_SIMPLE_MOD_BYPASS:-1}

ENTROPY_PENALTY_COEF=${ENTROPY_PENALTY_COEF:-0.03}

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is empty. Please use PyTorch DDP task type or set MASTER_ADDR manually."
  exit 1
fi

if [[ ! -f "${VISION_SD_QI_SCORING_FORMAT}" ]]; then
  echo "FATAL: VISION_SD_QI_SCORING_FORMAT is not a file: ${VISION_SD_QI_SCORING_FORMAT}"
  exit 1
fi

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "VISION_SD_TEACHER_B_REASONING_DELTA_MODE=${VISION_SD_TEACHER_B_REASONING_DELTA_MODE} VISION_SD_QI_SCORING_FORMAT=${VISION_SD_QI_SCORING_FORMAT}"
echo "worker.actor.entropy_penalty_coef=${ENTROPY_PENALTY_COEF}"

ray stop || true

if [[ "${NODE_RANK}" == "0" ]]; then
  ray start --head --node-ip-address="${MASTER_ADDR}" --port="${MASTER_PORT}"

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
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.entropy_penalty_coef=${ENTROPY_PENALTY_COEF} \
    worker.rollout.n=8 \
    worker.rollout.top_p=0.99 \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.online_filtering=true \
    trainer.total_epochs=2 \
    trainer.experiment_name=qwen2_5_vl_7b_simple_rlsd_proportional_rlsvr_rollout_qi_dapo_entpen${ENTROPY_PENALTY_COEF}_nosimplemod_${NNODES}n${NPROC_PER_NODE}g \
    trainer.save_checkpoint_path=/data/250010207/Vision-SR1/saves/7b_simple_rlsd_proportional_rlsvr_rollout_qi_dapo_entpen${ENTROPY_PENALTY_COEF}_nosimplemod_${NNODES}n${NPROC_PER_NODE}g \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    trainer.val_before_train=true \
    trainer.val_only=false
else
  ray start --address="${MASTER_ADDR}:${MASTER_PORT}" --block
fi
