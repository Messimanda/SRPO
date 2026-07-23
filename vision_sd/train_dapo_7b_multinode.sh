#!/bin/bash
#
# Multi-node: same recipe as train_dapo_3b_multinode.sh (DAPO-style: no KL, online_filtering, clip 0.2/0.28),
# for Qwen2.5-VL-7B-Instruct. Default: 2 nodes × 8 GPUs.
#
# Requires MASTER_ADDR. Rank 0: Ray head + vision_sd.main; other ranks: ray worker --block.
#
# Example:
#   MASTER_ADDR=10.0.0.1 bash vision_sd/train_dapo_7b_multinode.sh

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

export PYTHONUNBUFFERED=1
export VISION_SD_CREDIT_MODE=${VISION_SD_CREDIT_MODE:-pure_grpo}
export VISION_SD_RUNNER_NUM_GPUS=${VISION_SD_RUNNER_NUM_GPUS:-0}

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is empty. Set MASTER_ADDR to the head node IP (e.g. PyTorch DDP task)."
  exit 1
fi

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"

ray stop || true

if [[ "${NODE_RANK}" == "0" ]]; then
  ray start --head --node-ip-address="${MASTER_ADDR}" --port="${MASTER_PORT}"

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
    trainer.experiment_name=qwen2_5_vl_7b_dapo_baseline_${NNODES}n${NPROC_PER_NODE}g \
    trainer.save_checkpoint_path=/data/250010207/Vision-SR1/saves/7b_dapo_baseline_${NNODES}n${NPROC_PER_NODE}g \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    trainer.val_before_train=true \
    trainer.val_only=false
else
  ray start --address="${MASTER_ADDR}:${MASTER_PORT}" --block
fi
