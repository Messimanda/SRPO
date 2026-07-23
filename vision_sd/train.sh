#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
# MODEL_PATH=${1:-Qwen/Qwen2.5-VL-3B-Instruct}
MODEL_PATH=${1:-/data/250010207/Vision-SR1/LLaMA-Factory-Cold-Start/saves/sft_models/Vision-SD-Cold-Start}
# MODEL_PATH=${1:-/data/share/Qwen2.5-VL-7B-Instruct}
# MODEL_PATH=${1:-Qwen/Qwen3-VL-8B-Instruct}

export VISION_SD_TEACHER_A_MODEL=${VISION_SD_TEACHER_A_MODEL:-${MODEL_PATH}}
export VISION_SD_TEACHER_A_NUM_VIEWS=${VISION_SD_TEACHER_A_NUM_VIEWS:-2}
export VISION_SD_TEACHER_A_FALLBACK_ON_ERROR=${VISION_SD_TEACHER_A_FALLBACK_ON_ERROR:-1}

python3 -m vision_sd.main \
    config=vision_sd/config.yaml \
    data.train_files=/data/250010207/Vision-SR1/datasets/Vision-SR1-47K@train \
    data.val_files=/data/250010207/Vision-SR1/datasets/mmstar@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.experiment_name=qwen2_5_vl_7b_credit_distill_grpo \
    trainer.save_checkpoint_path=./saves/7b_grpo_credit_distill \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=true \
    trainer.val_only=false
