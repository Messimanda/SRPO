#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=${1:-/data/250010207/Vision-SR1/LLaMA-Factory-Cold-Start/saves/sft_models/Vision-SR1-Cold-Start}
TRAIN_FILES=${TRAIN_FILES:-/data/250010207/Vision-SR1/datasets/Vision-SR1-47K@train}
VAL_FILES=${VAL_FILES:-/data/250010207/Vision-SR1/datasets/mmstar@test}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
NUM_GPUS=${NUM_GPUS:-8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vision_sd_cold_start_pure_grpo}
SAVE_PATH=${SAVE_PATH:-./saves/vision_sd_cold_start_pure_grpo}

python3 -m vision_r1.main \
    config=vision_r1/config.yaml \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=${ROLLOUT_N} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.val_before_train=true \
    trainer.val_only=false
