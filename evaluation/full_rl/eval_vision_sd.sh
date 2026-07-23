#!/bin/bash
#
# Evaluate a full-finetuned vision_sd checkpoint or an exported Hugging Face model.
#
# Usage:
#   # 1) Evaluate the base model from vision_sd/train.sh
#   bash evaluation/full_rl/eval_vision_sd.sh
#
#   # 2) Evaluate an exported Hugging Face model directory directly
#   bash evaluation/full_rl/eval_vision_sd.sh \
#     /data/250010207/Vision-SR1/saves/7b_grpo_credit_distill/global_step_80/actor/huggingface
#
#   # 3) Evaluate a training checkpoint directory
#   bash evaluation/full_rl/eval_vision_sd.sh \
#     /data/250010207/Vision-SR1/saves/7b_grpo_credit_distill/global_step_80 \
#     /data/share/Qwen2.5-VL-7B-Instruct
#
#   # 4) Override the local judge model path if needed
#   bash evaluation/full_rl/eval_vision_sd.sh \
#     /data/250010207/Vision-SR1/saves/7b_grpo_credit_distill/global_step_80/actor/huggingface \
#     /data/share/Qwen2.5-VL-7B-Instruct \
#     /data/share/Qwen2.5-14B-Instruct
#
#   # 5) Use GPT-4o as the LLM judge
#   OPENAI_API_KEY=your_key \
#   bash evaluation/full_rl/eval_vision_sd.sh \
#     /data/250010207/Vision-SR1/saves/7b_grpo_credit_distill/global_step_80/actor/huggingface \
#     /data/share/Qwen2.5-VL-7B-Instruct \
#     gpt-4o \
#     openai
#

set -x
set -euo pipefail
export PYTHONUNBUFFERED=1

INPUT_PATH="${1:-}"
DEFAULT_BASE_MODEL="/data/share/Qwen2.5-VL-7B-Instruct"
DEFAULT_JUDGE_MODEL="/data/share/Qwen2.5-VL-72B-Instruct"
LOCAL_EVAL_ROOT="/data/share/eval_local"
LOCAL_TRAIN_DATASET="${LOCAL_EVAL_ROOT}/geometry3k/data"

CHECKPOINT_PATH=""
MODEL_PATH="${2:-$DEFAULT_BASE_MODEL}"
JUDGE_MODEL="${3:-$DEFAULT_JUDGE_MODEL}"
JUDGE_BACKEND="${4:-auto}"

if [ -n "${INPUT_PATH}" ]; then
  if [ -d "${INPUT_PATH}" ] && [ -f "${INPUT_PATH}/config.json" ]; then
    MODEL_PATH="${INPUT_PATH}"
  else
    CHECKPOINT_PATH="${INPUT_PATH}"
  fi
fi

MODEL_NAME="${MODEL_PATH##*/}"
SAVE_DIR="./evaluation/responses/vision_sd_full_change/${MODEL_NAME}"

DATASETS=(
  "${LOCAL_EVAL_ROOT}/mmstar"
  "${LOCAL_EVAL_ROOT}/mm-vet"
  "${LOCAL_EVAL_ROOT}/MLLM_test"
  "${LOCAL_EVAL_ROOT}/visnumbench"
  "${LOCAL_EVAL_ROOT}/mmmu_pro_10options"
  "${LOCAL_EVAL_ROOT}/mmmu-pro-vision"
  "${LOCAL_EVAL_ROOT}/hallusionbench"
  "${LOCAL_EVAL_ROOT}/MMMU"
  "${LOCAL_EVAL_ROOT}/MMSI"
  "${LOCAL_EVAL_ROOT}/mathverse"
  "${LOCAL_EVAL_ROOT}/mathvision"
  "${LOCAL_EVAL_ROOT}/mathvista"
  "${LOCAL_EVAL_ROOT}/realWorldQA"
)

if [ ! -d "${LOCAL_TRAIN_DATASET}" ]; then
  echo "Missing local train placeholder dataset: ${LOCAL_TRAIN_DATASET}"
  exit 1
fi

for DS in "${DATASETS[@]}"; do
  if [ ! -d "${DS}" ]; then
    echo "Missing local eval dataset: ${DS}"
    exit 1
  fi
done

CKPT_ARGS=""
if [ -n "${CHECKPOINT_PATH}" ]; then
  CKPT_ARGS="trainer.load_checkpoint_path=${CHECKPOINT_PATH}"
  STEP_NAME="${CHECKPOINT_PATH##*/}"
  SAVE_DIR="./evaluation/responses/vision_sd_full_change/${MODEL_NAME}/${STEP_NAME}"
fi

for DS in "${DATASETS[@]}"; do
  SHORT_NAME="${DS##*/}"
  RESPONSE_PATH="${SAVE_DIR}/${SHORT_NAME}.jsonl"

  if [ -s "${RESPONSE_PATH}" ]; then
    echo ">>> Skipping ${SHORT_NAME}; existing responses found at ${RESPONSE_PATH}"
    echo "------------------------------------------------------------"
    continue
  fi

  echo ">>> Evaluating ${SHORT_NAME} with vision_sd"

  python3 -m vision_sd.main \
    config=vision_sd/config.yaml \
    data.train_files="${LOCAL_TRAIN_DATASET}@test" \
    data.val_files="${DS}@test" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.format_prompt=./vision_sd/format_prompt/perception_reasoning_answer.jinja \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.rollout.n=1 \
    worker.rollout.temperature=0.0 \
    worker.rollout.top_p=1.0 \
    worker.reward.reward_function=./vision_sd/reward_function/answer_reward.py:compute_score \
    trainer.experiment_name="eval_vision_sd_full_change_${SHORT_NAME}" \
    trainer.response_path="${RESPONSE_PATH}" \
    trainer.n_gpus_per_node=4 \
    trainer.find_last_checkpoint=false \
    trainer.val_only=true \
    ${CKPT_ARGS}

  echo "------------------------------------------------------------"
done

echo "All responses saved under: ${SAVE_DIR}/"

# --- LLM Judge (extract \boxed{} then judge with LLM) ---
JUDGMENT_DIR="./evaluation/judgments/${SAVE_DIR#./evaluation/responses/}"
echo ">>> Running LLM Judge..."
python evaluation/llm_judge.py \
  --response_dir "${SAVE_DIR}" \
  --judge_model "${JUDGE_MODEL}" \
  --judge_backend "${JUDGE_BACKEND}"
python evaluation/print_accuracy.py --judgment_dir "${JUDGMENT_DIR}"
