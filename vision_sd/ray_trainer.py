"""
Gated dual-teacher credit distillation trainer.

This trainer keeps the RL reward as the optimization direction and uses
teacher branches to reshape token-level credit:
1. Teacher A (optional / mode-dependent): external counterfactual support estimation over perception evidence.
2. Teacher B: self privileged-vs-base discrepancy for reasoning grounding.
3. ``simple_rlsd`` mode: perception magnitude from stop-grad Δ log p (real vs partially masked image, same prompt), no Teacher A.
4. With ``VISION_SD_SIMPLE_TEACHER_B_MOD=rlsvr``, Teacher B still compares privileged **with-perception** forward vs rollout on the **reasoning** span, but token credit on reasoning follows RLVR (w=exp(sign(A)Δ), clip) instead of z-norm+gate+``simple_mod`` on that span.
5. With ``VISION_SD_TEACHER_B_GROUND_TRUTH_PRIVILEGE`` (``simple_rlsd`` only), Teacher B can see the dataset ``ground_truth`` in its prompt and matches log-prob on the post-perception span (reasoning + answer); rows with empty ground truth use the original reasoning-only Teacher B.
6. With ``VISION_SD_TEACHER_B_REASONING_DELTA_MODE=rollout_vs_qi``, reasoning Δ is rollout log p minus a q+I-only rescore of the same tail; default ``with_p`` keeps the original Teacher B (privileged prompt vs student LPs).
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import ray
import torch
from jinja2 import Template
from ray.experimental.tqdm_ray import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.metrics import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ray_trainer import RayPPOTrainer, apply_kl_penalty
from verl.utils import torch_functional as VF
from PIL import Image as PILImage

from verl.utils.dataset import process_image
from verl.utils.logger import Tracker
from verl.utils.py_functional import convert_dict_to_str, timer, unflatten_dict


# Pre-exp clamp only to avoid inf; tunable credit bound is VISION_SD_RLSD_W_CLIP_* on w after exp.
_RLSD_LOG_RATIO_EXP_SANITY = 40.0

STRUCTURED_PATTERN = re.compile(
    r"^\s*(<perception>[\s\S]*?</perception>)\s*(<reasoning>[\s\S]*?</reasoning>)\s*(\\boxed\{[\s\S]*\})\s*$",
    re.DOTALL,
)
PERCEPTION_PATTERN = re.compile(r"<perception>([\s\S]*?)</perception>", re.DOTALL)
REASONING_PATTERN = re.compile(r"<reasoning>([\s\S]*?)</reasoning>", re.DOTALL)

TEACHER_A_SUPPORT_PROMPTS = [
    (
        "You are estimating evidence support, not judging writing quality.\n"
        "Question: {question}\n\n"
        "Perception evidence:\n{perception}\n\n"
        "Reasoning:\n{reasoning}\n\n"
        "Final answer:\n{answer}\n\n"
        "Decide how well the perception evidence supports the given reasoning and final answer. "
        "Return strict JSON with exactly these keys: support_score, reasoning_supported, "
        "answer_still_justified, missing_critical_evidence. "
        "support_score must be a float in [0,1]. The other keys must be booleans. "
        "Do not explain.\n"
        'Example: {{"support_score": 0.82, "reasoning_supported": true, '
        '"answer_still_justified": true, "missing_critical_evidence": false}}'
    ),
    (
        "Estimate counterfactual support for the current solution.\n"
        "Question: {question}\n"
        "Perception evidence units:\n{perception}\n"
        "Reasoning trace:\n{reasoning}\n"
        "Answer: {answer}\n\n"
        "Return only JSON with keys support_score, reasoning_supported, "
        "answer_still_justified, missing_critical_evidence. "
        "Score whether the answer remains supported by the evidence shown here, not whether the perception is well written."
    ),
    (
        "Given a question, evidence-only perception, a reasoning block, and a final answer, "
        "measure whether this evidence still supports the solution.\n\n"
        "Question: {question}\n\nEvidence:\n{perception}\n\nReasoning:\n{reasoning}\n\nAnswer:\n{answer}\n\n"
        "Output strict JSON with exactly four fields: support_score (float in [0,1]), "
        "reasoning_supported (bool), answer_still_justified (bool), missing_critical_evidence (bool)."
    ),
]

TEACHER_A_QP_QUALITY_PROMPTS = [
    (
        "Evaluate the evidence quality of the perception only.\n"
        "Question: {question}\n\n"
        "Perception evidence:\n{perception}\n\n"
        "Do not judge style. Return strict JSON with exactly one key: evidence_quality. "
        "evidence_quality must be a float in [0,1], where 1 means highly reliable and sufficient evidence."
    ),
    (
        "Assess how reliable and sufficient the perception evidence is for answering the question.\n"
        "Question: {question}\n"
        "Perception evidence units:\n{perception}\n\n"
        "Return only JSON: {\"evidence_quality\": <float in [0,1]>}."
    ),
    (
        "Given question and evidence-only perception, score evidence quality.\n"
        "Question: {question}\n\nEvidence:\n{perception}\n\n"
        "Output strict JSON with exactly one field: evidence_quality (float in [0,1])."
    ),
]

TEACHER_B_WITH_P_PROMPT = (
    "Question: {question}\n"
    "Perception:\n{perception}\n\n"
    "Continue with only the grounded reasoning block. "
    "Output exactly one <reasoning>...</reasoning> block."
)

# simple_rlsd + VISION_SD_TEACHER_B_GROUND_TRUTH_PRIVILEGE: Teacher B sees the labeled answer (privilege) and
# matches log p on the same post-perception token span as the rollout (reasoning + \\boxed{...}).
TEACHER_B_WITH_P_GT_ANSWER_PROMPT = (
    "Question: {question}\n"
    "Perception:\n{perception}\n\n"
    "Ground truth answer: {ground_truth}\n\n"
    "Continue with the same format as the student rollout: one <reasoning>...</reasoning> block "
    "followed by the final answer in \\boxed{{}}."
)

TEACHER_A_SIMPLE_EXT_QUALITY_PROMPTS = [
    (
        "Evaluate perception evidence quality for solving the question.\n"
        "Question: {question}\n\n"
        "Perception:\n{perception}\n\n"
        "Return strict JSON with exactly keys: relevance, factuality, leakage_penalty.\n"
        "Each field is a float in [0,1].\n"
        "- relevance: how relevant the perception is to the question.\n"
        "- factuality: how non-speculative and observation-grounded the perception is.\n"
        "- leakage_penalty: how much the perception leaks final conclusion/answer.\n"
        "Do not include explanations."
    ),
]

TEACHER_A_SIMPLE_SOLVE_PROMPTS = [
    (
        "Assess whether the perception evidence alone is sufficient to solve the question.\n"
        "Question: {question}\n\n"
        "Perception:\n{perception}\n\n"
        "Return strict JSON with exactly one key: solve_score (float in [0,1])."
    ),
]


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def grpo_advantage_for_scores(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    eps: float = 1e-6,
) -> torch.Tensor:
    id2score = defaultdict(list)
    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])

    id2mean, id2std = {}, {}
    for idx, vals in id2score.items():
        stacked = torch.stack(vals)
        id2mean[idx] = stacked.mean()
        id2std[idx] = stacked.std()

    normed = scores.clone()
    for i in range(scores.shape[0]):
        normed[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    return normed.unsqueeze(-1) * response_mask


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return float(min(1.0, max(0.0, float(value))))
    except Exception:
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1"}:
            return True
        if value.lower() in {"false", "no", "0"}:
            return False
    return default


def split_evidence_units(text: str) -> list[str]:
    """Heuristic evidence-unit segmentation for perception text."""
    raw_segments = [seg.strip() for seg in re.split(r"\n+|(?<=[。；;])\s*|(?<=\.)\s+(?=[A-Z0-9])", text) if seg.strip()]
    refined: list[str] = []
    splitters = re.compile(r"\s*(?:, and | and |, plus | plus |, with | with |, while | while |, showing | showing )\s*")

    for segment in raw_segments:
        parts = [part.strip(" ,;") for part in splitters.split(segment) if part.strip(" ,;")]
        for part in parts:
            if len(part.split()) > 18 and "," in part:
                refined.extend([sub.strip(" ,;") for sub in part.split(",") if sub.strip(" ,;")])
            else:
                refined.append(part)

    refined = [piece for piece in refined if piece]
    return refined if refined else [text.strip()] if text.strip() else []


def normalize_nonnegative(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.array([max(0.0, float(v)) for v in values], dtype=np.float32)
    if float(array.max()) <= 1e-6:
        return [0.0 for _ in values]
    return (array / (float(array.max()) + 1e-6)).tolist()


def parse_structured_response(text: str) -> Optional[dict[str, str]]:
    match = STRUCTURED_PATTERN.match(text.strip())
    if not match:
        return None

    perception_block, reasoning_block, answer_block = match.groups()
    perception_match = PERCEPTION_PATTERN.search(perception_block)
    reasoning_match = REASONING_PATTERN.search(reasoning_block)
    if perception_match is None or reasoning_match is None:
        return None

    return {
        "perception_block": perception_block.strip(),
        "reasoning_block": reasoning_block.strip(),
        "answer_block": answer_block.strip(),
        "perception_text": perception_match.group(1).strip(),
        "reasoning_text": reasoning_match.group(1).strip(),
    }


class CreditDistillTrainer(RayPPOTrainer):
    """RL trainer with dual-teacher token-level credit assignment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit_mode = os.getenv("VISION_SD_CREDIT_MODE", "full").strip().lower()
        # Teacher A backend:
        # - "vllm": standalone vLLM engine (default legacy behavior)
        # - "actor": reuse the training actor/rollout model for teacher generation
        self.teacher_a_backend = os.getenv("VISION_SD_TEACHER_A_BACKEND", "vllm").strip().lower()
        self.teacher_a_model_path = os.getenv("VISION_SD_TEACHER_A_MODEL", "/data/share/Qwen2.5-VL-72B-Instruct")
        self.teacher_a_disabled = self.teacher_a_model_path.strip().lower() in {"", "none", "off", "disabled"}
        self.teacher_a_fallback_on_error = _bool_env("VISION_SD_TEACHER_A_FALLBACK_ON_ERROR", True)
        self.teacher_a_temperature = _float_env("VISION_SD_TEACHER_A_TEMPERATURE", 0.1)
        self.teacher_a_max_tokens = int(os.getenv("VISION_SD_TEACHER_A_MAX_TOKENS", "128"))
        self.teacher_a_num_views = max(1, int(os.getenv("VISION_SD_TEACHER_A_NUM_VIEWS", "2")))
        self.teacher_a_conf_tau = _float_env("VISION_SD_TEACHER_A_CONF_TAU", 4.0)
        self.teacher_a_floor = _float_env("VISION_SD_TEACHER_A_FLOOR", 0.3)
        self.teacher_a_support_coef = _float_env("VISION_SD_TEACHER_A_SUPPORT_COEF", 0.55)
        self.teacher_a_reasoning_coef = _float_env("VISION_SD_TEACHER_A_REASONING_COEF", 0.20)
        self.teacher_a_answer_coef = _float_env("VISION_SD_TEACHER_A_ANSWER_COEF", 0.20)
        self.teacher_a_missing_coef = _float_env("VISION_SD_TEACHER_A_MISSING_COEF", 0.15)
        self.teacher_a_use_qp_quality = _bool_env("VISION_SD_TEACHER_A_USE_QP_QUALITY", True)
        self.teacher_a_quality_blend = clamp01(_float_env("VISION_SD_TEACHER_A_QUALITY_BLEND", 0.5), default=0.5)
        self.teacher_a_quality_calibration = _bool_env("VISION_SD_TEACHER_A_QUALITY_CALIBRATION", True)
        self.teacher_a_quality_calib_momentum = clamp01(
            _float_env("VISION_SD_TEACHER_A_QUALITY_CALIB_MOMENTUM", 0.95),
            default=0.95,
        )
        self.teacher_a_quality_calib_temp = max(0.1, _float_env("VISION_SD_TEACHER_A_QUALITY_CALIB_TEMP", 1.0))
        self.teacher_a_quality_calib_eps = max(1e-6, _float_env("VISION_SD_TEACHER_A_QUALITY_CALIB_EPS", 1e-4))
        self.teacher_b_floor = _float_env("VISION_SD_TEACHER_B_FLOOR", 0.2)
        self.reasoning_gate_beta = _float_env("VISION_SD_REASONING_GATE_BETA", 8.0)
        self.reasoning_gate_delta = _float_env("VISION_SD_REASONING_GATE_DELTA", 0.35)
        self.reasoning_gate_reject_delta = _float_env("VISION_SD_REASONING_GATE_REJECT_DELTA", 0.2)
        self.reasoning_gate_margin = max(0.0, _float_env("VISION_SD_REASONING_GATE_MARGIN", 0.05))
        self.reasoning_neg_adv_alpha = clamp01(_float_env("VISION_SD_REASONING_NEG_ADV_ALPHA", 0.2), default=0.2)
        self.max_perception_spans = max(1, int(os.getenv("VISION_SD_MAX_PERCEPTION_SPANS", "12")))
        self.simple_ext_rel_coef = _float_env("VISION_SD_SIMPLE_EXT_REL_COEF", 0.5)
        self.simple_ext_fact_coef = _float_env("VISION_SD_SIMPLE_EXT_FACT_COEF", 0.5)
        self.simple_ext_leak_coef = _float_env("VISION_SD_SIMPLE_EXT_LEAK_COEF", 0.35)
        self.simple_gate_lambda_ext = _float_env("VISION_SD_SIMPLE_GATE_LAMBDA_EXT", 0.6)
        self.simple_gate_lambda_solve = _float_env("VISION_SD_SIMPLE_GATE_LAMBDA_SOLVE", 0.4)
        # If True: skip Teacher-A solve prompts; q_p uses q_ext only (no q_solve mix).
        self.simple_q_ext_only = _bool_env("VISION_SD_SIMPLE_Q_EXT_ONLY", False)
        self.simple_gate_delta = _float_env("VISION_SD_SIMPLE_GATE_DELTA", 0.35)
        self.simple_gate_beta = _float_env("VISION_SD_SIMPLE_GATE_BETA", 8.0)
        self.simple_perception_floor = _float_env("VISION_SD_SIMPLE_PERCEPTION_FLOOR", 0.35)
        self.simple_reasoning_support_scale = _float_env("VISION_SD_SIMPLE_REASONING_SUPPORT_SCALE", 0.6)
        self.simple_mod_lambda = _float_env("VISION_SD_SIMPLE_MOD_LAMBDA", 0.5)
        self.simple_mod_clip = _float_env("VISION_SD_SIMPLE_MOD_CLIP", 0.4)
        self.simple_mod_mult_min = _float_env("VISION_SD_SIMPLE_MOD_MULT_MIN", 0.3)
        self.simple_mod_mult_max = _float_env("VISION_SD_SIMPLE_MOD_MULT_MAX", 1.7)
        # If True: use raw_w as modulation (no mean-center+clip+re-scale). rlsvr still overwrites reasoning span.
        self.simple_mod_bypass = _bool_env("VISION_SD_SIMPLE_MOD_BYPASS", False)
        # If True and teacher_b_mod=rlsvr: write rlsvr reasoning weights into raw_w first, then run one unified simple_mod.
        # If False: keep legacy behavior (run simple_mod first, then overwrite reasoning span with rlsvr weights).
        self.rlsvr_before_simple_mod = _bool_env("VISION_SD_RLSVR_BEFORE_SIMPLE_MOD", False)
        self.simple_utility_mode = os.getenv("VISION_SD_SIMPLE_UTILITY_MODE", "topk_mask").strip().lower()
        self.simple_utility_topk = max(1, int(os.getenv("VISION_SD_SIMPLE_UTILITY_TOPK", "3")))
        self.simple_utility_drop_ratio = min(0.8, max(0.1, _float_env("VISION_SD_SIMPLE_UTILITY_DROP_RATIO", 0.3)))
        self.simple_utility_placeholder = os.getenv(
            "VISION_SD_SIMPLE_UTILITY_PLACEHOLDER", "[MISSING_KEY_EVIDENCE]"
        ).strip()
        # RLSD-style perception credit: strong = real vision; weak = same prompt + masked images (or text_only).
        self.rlsd_weak_mode = os.getenv("VISION_SD_RLSD_WEAK_MODE", "mask_image").strip().lower()
        self.rlsd_image_mask_ratio = clamp01(_float_env("VISION_SD_RLSD_IMAGE_MASK_RATIO", 0.5), default=0.5)
        self.rlsd_image_mask_style = os.getenv("VISION_SD_RLSD_IMAGE_MASK_STYLE", "pixel").strip().lower()
        self.rlsd_image_mask_grid_h = max(2, int(os.getenv("VISION_SD_RLSD_IMAGE_MASK_GRID_H", "8")))
        self.rlsd_image_mask_grid_w = max(2, int(os.getenv("VISION_SD_RLSD_IMAGE_MASK_GRID_W", "8")))
        # Fixed-size random patch blackening (non-overlapping tiles); used when style is random_patch / patch_blacken / rpb.
        self.rlsd_image_mask_patch_h = max(1, int(os.getenv("VISION_SD_RLSD_PATCH_H", "14")))
        self.rlsd_image_mask_patch_w = max(1, int(os.getenv("VISION_SD_RLSD_PATCH_W", "14")))
        self.rlsd_lambda_warmup_steps = int(os.getenv("VISION_SD_RLSD_LAMBDA_WARMUP_STEPS", "0"))
        self.rlsd_gate_k = _float_env("VISION_SD_RLSD_GATE_K", 6.0)
        # simple_rlsd Teacher-B gate only: add to lr_m before σ(β·(k·lr_m′−δ)). Does not change perception w.
        self.simple_rlsd_gate_lr_m_bias = _float_env("VISION_SD_SIMPLE_RLSD_GATE_LR_M_BIAS", 0.0)
        # If set (e.g. 1.0), Teacher-B gate ignores lr_m and uses this value in [0,1] (full-strength modulation).
        _tb_gate_fixed = os.getenv("VISION_SD_SIMPLE_RLSD_TEACHER_B_GATE_FIXED")
        if _tb_gate_fixed is None or str(_tb_gate_fixed).strip() == "":
            self.simple_rlsd_teacher_b_gate_fixed: Optional[float] = None
        else:
            self.simple_rlsd_teacher_b_gate_fixed = float(_tb_gate_fixed)
        self.rlsd_normalize_per_seq = _bool_env("VISION_SD_RLSD_NORMALIZE_PER_SEQ", True)
        _wlo = _float_env("VISION_SD_RLSD_W_CLIP_LOW", 0.8)
        _whi = _float_env("VISION_SD_RLSD_W_CLIP_HIGH", 1.2)
        if _wlo < _whi and _wlo > 0.0:
            self.rlsd_w_clip_low, self.rlsd_w_clip_high = float(_wlo), float(_whi)
        else:
            self.rlsd_w_clip_low, self.rlsd_w_clip_high = 0.8, 1.2
        # Perception-only token credit from Δ log p (same prompt, original image vs masked):
        # - "exp": w_t = exp(clamp(log_strong - log_weak)); optional per-seq mean norm + clip (default).
        # - "proportional": w_t ∝ exp(Δ_t), per-seq mean norm; no post-hoc w_clip (likelihood-ratio weights).
        # - "softmax": w = softmax(β·Δ) * T_perc so sum_t w_t = T_perc (fixed total mass on perception span).
        self.rlsd_perception_credit_alloc = os.getenv(
            "VISION_SD_RLSD_PERCEPTION_CREDIT_ALLOC", "exp"
        ).strip().lower()
        self.rlsd_softmax_beta = _float_env("VISION_SD_RLSD_SOFTMAX_BETA", 1.0)
        # Reward shaping for simple_rlsd: R' = R + λ·max(0, Δ_mean) with Δ_mean = mean_t(log_strong - log_weak) on perception.
        self.rlsd_reward_delta_lambda = _float_env("VISION_SD_RLSD_REWARD_DELTA_LAMBDA", 0.0)
        # simple_rlsd / Teacher B: "legacy" or "rlsvr" (RLVR-style modulation on the reasoning span only).
        self.simple_teacher_b_mod = os.getenv("VISION_SD_SIMPLE_TEACHER_B_MOD", "legacy").strip().lower()
        self.teacher_b_paper_w_eps = _float_env("VISION_SD_TEACHER_B_PAPER_W_EPS", 0.2)
        self.teacher_b_paper_delta_clip = _float_env("VISION_SD_TEACHER_B_PAPER_DELTA_CLIP", 20.0)
        # If True (simple_rlsd only): Teacher B uses dataset ground_truth in the teacher prompt and distills
        # the post-perception span (reasoning + answer tokens), not reasoning-only. Rows with empty ground_truth
        # fall back to the original Teacher B (reasoning-only) behavior.
        self.teacher_b_gt_privilege = _bool_env("VISION_SD_TEACHER_B_GROUND_TRUTH_PRIVILEGE", False)
        # Reasoning Δ for Teacher B / rlsvr: "with_p" = TEACHER_B prompt vs student LPs (default);
        # "rollout_vs_qi" = student rollout LPs vs q+I-only rescore of the same tail (no extra with_p forward).
        self.teacher_b_reasoning_delta_mode = os.getenv(
            "VISION_SD_TEACHER_B_REASONING_DELTA_MODE", "with_p"
        ).strip().lower()
        self._rollout_format_prompt_source: Optional[str] = None
        self._qi_scoring_format_source: Optional[str] = None
        self._teacher_a_tokenizer = None
        self._teacher_a_llm = None
        self._teacher_a_sampling_params = None
        self._teacher_a_init_error = None
        self._teacher_a_quality_stats = {
            "support": {"mean": None, "sq_mean": None},
            "evidence": {"mean": None, "sq_mean": None},
        }
        self._simple_rng = np.random.default_rng(int(os.getenv("VISION_SD_SIMPLE_RANDOM_SEED", "1")))

    def _build_messages(self, prompt_text: str, use_image: bool, image_count: int = 0) -> list[dict[str, Any]]:
        if use_image and image_count > 0:
            content_list = []
            for i, content in enumerate(prompt_text.split("<image>")):
                if 0 < i <= image_count:
                    content_list.append({"type": "image"})
                if content:
                    content_list.append({"type": "text", "text": content})
            for _ in range(max(0, image_count - prompt_text.count("<image>"))):
                content_list.append({"type": "image"})
            return [{"role": "user", "content": content_list}]
        return [{"role": "user", "content": prompt_text}]

    def _encode_prompt(self, prompt_text: str, multi_modal_data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        images = multi_modal_data.get("images", []) if multi_modal_data is not None else []
        use_image = len(images) > 0
        messages = self._build_messages(prompt_text, use_image=use_image, image_count=len(images))

        if use_image:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images = [process_image(image, self.config.data.min_pixels, self.config.data.max_pixels) for image in images]
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and hasattr(self.processor, "image_processor") and (
            "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": np.array(raw_prompt_ids, dtype=object),
            "multi_modal_data": multi_modal_data,
        }

    def _build_prompt_batch(
        self,
        prompts: list[str],
        multi_modal_batch: Optional[list[Optional[dict[str, Any]]]] = None,
        uids: Optional[list[Any]] = None,
    ) -> DataProto:
        encoded = []
        if multi_modal_batch is None:
            multi_modal_batch = [None] * len(prompts)
        if uids is None:
            uids = list(range(len(prompts)))

        for prompt, multi_modal_data in zip(prompts, multi_modal_batch):
            encoded.append(self._encode_prompt(prompt, multi_modal_data=multi_modal_data))

        tensors = {
            "input_ids": torch.stack([item["input_ids"] for item in encoded]),
            "attention_mask": torch.stack([item["attention_mask"] for item in encoded]),
            "position_ids": torch.stack([item["position_ids"] for item in encoded]),
        }
        non_tensors = {
            "raw_prompt_ids": np.array([item["raw_prompt_ids"] for item in encoded], dtype=object),
        }
        if any(item["multi_modal_data"] is not None for item in encoded):
            non_tensors["multi_modal_data"] = np.array([item["multi_modal_data"] for item in encoded], dtype=object)
            non_tensors["uid"] = np.array(uids, dtype=object)

        meta_info = {
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "min_pixels": self.config.data.min_pixels,
            "max_pixels": self.config.data.max_pixels,
            "video_fps": self.config.data.video_fps,
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def _build_response_tensors(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        model_inputs = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_response_length,
            return_tensors="pt",
        )
        return model_inputs["input_ids"], model_inputs["attention_mask"].to(torch.long)

    def _load_rollout_format_prompt_source(self) -> Optional[str]:
        if self._rollout_format_prompt_source is not None:
            return self._rollout_format_prompt_source or None
        path = getattr(self.config.data, "format_prompt", None)
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                self._rollout_format_prompt_source = f.read()
        else:
            self._rollout_format_prompt_source = ""
        return self._rollout_format_prompt_source or None

    def _render_rollout_prompt_text(self, raw_question: str, multi_modal_data: Optional[dict[str, Any]]) -> str:
        """Rebuild the same user prompt string as RLHFDataset (format jinja + optional <image>/<video>)."""
        prompt_str = raw_question
        fmt = self._load_rollout_format_prompt_source()
        if fmt:
            prompt_str = Template(fmt.strip()).render(content=prompt_str)

        has_images = bool(multi_modal_data and multi_modal_data.get("images"))
        has_videos = bool(multi_modal_data and multi_modal_data.get("videos"))
        if has_images and "<image>" not in prompt_str:
            prompt_str = prompt_str + "<image>"
        if has_videos and "<video>" not in prompt_str:
            prompt_str = prompt_str + "<video>"
        return prompt_str

    def _load_qi_scoring_format_source(self) -> str:
        """Jinja for q+I-only rescoring: no <perception> block; see ``format_prompt/qi_only_reasoning_answer.jinja``."""
        if self._qi_scoring_format_source is not None:
            return self._qi_scoring_format_source
        path = os.getenv("VISION_SD_QI_SCORING_FORMAT", "").strip()
        if not path:
            path = os.path.join(os.path.dirname(__file__), "format_prompt", "qi_only_reasoning_answer.jinja")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                self._qi_scoring_format_source = f.read()
        else:
            self._qi_scoring_format_source = (
                "{{ content | trim }} Analyze the image/video and solve the problem. "
                "Write the reasoning process grounded in the image/video, wrapped in <reasoning> </reasoning> tags. "
                "Finally, give the answer in \\boxed{}. The required format is: "
                "<reasoning> grounded reasoning here </reasoning> \\boxed{FINAL ANSWER}."
            )
        return self._qi_scoring_format_source

    def _render_scoring_prompt_qi_only(self, raw_question: str, multi_modal_data: Optional[dict[str, Any]]) -> str:
        """User prompt for counterfactual log p(y_tail | q, I) — same image/video rules as ``_render_rollout_prompt_text``."""
        prompt_str = raw_question
        fmt = self._load_qi_scoring_format_source()
        if fmt:
            prompt_str = Template(fmt.strip()).render(content=prompt_str)
        has_images = bool(multi_modal_data and multi_modal_data.get("images"))
        has_videos = bool(multi_modal_data and multi_modal_data.get("videos"))
        if has_images and "<image>" not in prompt_str:
            prompt_str = prompt_str + "<image>"
        if has_videos and "<video>" not in prompt_str:
            prompt_str = prompt_str + "<video>"
        return prompt_str

    def _weak_prompt_remove_visual_placeholders(self, prompt_text: str) -> str:
        """Text-only counterfactual prompt: drop explicit vision placeholders, keep instructions."""
        text = re.sub(r"<image>\s*", "", prompt_text)
        text = re.sub(r"<video>\s*", "", text)
        return text.strip()

    def _apply_rlsd_image_mask(self, pil: PILImage.Image) -> PILImage.Image:
        """Weak-branch image: blacken by pixel noise, fixed-size random patches, or equal-split grid. ratio=1 => all black."""
        r = float(self.rlsd_image_mask_ratio)
        r = max(0.0, min(1.0, r))
        if r >= 1.0 - 1e-6:
            return PILImage.new("RGB", pil.size, (0, 0, 0))
        if r <= 1e-6:
            return pil
        arr = np.asarray(pil.convert("RGB"), dtype=np.uint8).copy()
        h, w = arr.shape[0], arr.shape[1]
        style = self.rlsd_image_mask_style
        if style in {"grid", "patch"}:
            gh, gw = self.rlsd_image_mask_grid_h, self.rlsd_image_mask_grid_w
            for i in range(gh):
                y0 = i * h // gh
                y1 = (i + 1) * h // gh if i < gh - 1 else h
                for j in range(gw):
                    if float(self._simple_rng.random()) < r:
                        x0 = j * w // gw
                        x1 = (j + 1) * w // gw if j < gw - 1 else w
                        arr[y0:y1, x0:x1] = 0
        elif style in {"random_patch", "patch_blacken", "rpb"}:
            ph, pw = self.rlsd_image_mask_patch_h, self.rlsd_image_mask_patch_w
            for y0 in range(0, h, ph):
                for x0 in range(0, w, pw):
                    if float(self._simple_rng.random()) < r:
                        y1 = min(y0 + ph, h)
                        x1 = min(x0 + pw, w)
                        arr[y0:y1, x0:x1] = 0
        else:
            draw = self._simple_rng.random((h, w)) < r
            arr[draw] = 0
        return PILImage.fromarray(arr)

    def _rlsd_masked_images_multi_modal(self, mm: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Same template as strong; images are process_image'd then partially or fully masked."""
        if mm is None or not isinstance(mm, dict) or not mm.get("images"):
            return None
        masked_images: list[Any] = []
        for img in mm["images"]:
            pil = process_image(img, self.config.data.min_pixels, self.config.data.max_pixels)
            masked_images.append(self._apply_rlsd_image_mask(pil))
        weak_mm: dict[str, Any] = {"images": masked_images}
        if mm.get("videos"):
            weak_mm["videos"] = list(mm["videos"])
        return weak_mm

    def _compute_perception_image_text_logprob_diff(
        self,
        *,
        valid_indices: list[int],
        questions: list[str],
        response_ids: torch.Tensor,
        response_lengths: torch.Tensor,
        parsed_samples: list[Optional[dict[str, str]]],
        multi_modal_batch: np.ndarray,
        valid_uids: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two stop-grad forwards on the *perception* token span only.

        - Strong: same rendered user prompt + original ``multi_modal_data`` pixels.
        - Weak (``mask_image``): same prompt string + masked images (pixel, fixed random patches, or grid; see
          ``_apply_rlsd_image_mask`` / ``_rlsd_masked_images_multi_modal``).
        Returns per-token ``log p(y_t | prompt, I)`` and ``log p(y_t | prompt, I')`` for t in the
        perception prefix; downstream credit uses ``log_strong - log_weak`` at token resolution.
        """
        strong_prompts: list[str] = []
        weak_prompts: list[str] = []
        mm_strong: list[Optional[dict[str, Any]]] = []
        mm_weak: list[Optional[dict[str, Any]]] = []
        perception_slices: list[torch.Tensor] = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        for batch_idx in valid_indices:
            q = questions[batch_idx]
            parsed = parsed_samples[batch_idx]
            assert parsed is not None
            cur_len = int(response_lengths[batch_idx].item())
            perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            perception_len = min(perception_len, cur_len)
            perception_slices.append(response_ids[batch_idx, :perception_len].long().cpu())

            mm = multi_modal_batch[batch_idx]
            if mm is not None and not isinstance(mm, dict):
                mm = None
            rendered = self._render_rollout_prompt_text(q, mm)
            strong_prompts.append(rendered)
            mm_strong.append(mm)

            if self.rlsd_weak_mode in {"mask_image", "mask", "masked"}:
                wmm = self._rlsd_masked_images_multi_modal(mm)
                if wmm is not None:
                    weak_prompts.append(rendered)
                    mm_weak.append(wmm)
                else:
                    weak_prompts.append(self._weak_prompt_remove_visual_placeholders(rendered))
                    mm_weak.append(None)
            else:
                weak_prompts.append(self._weak_prompt_remove_visual_placeholders(rendered))
                mm_weak.append(None)

        responses = pad_sequence(perception_slices, batch_first=True, padding_value=pad_id)
        plens = torch.tensor([t.numel() for t in perception_slices], dtype=torch.long)
        max_plen = responses.size(1)
        response_mask = (torch.arange(max_plen).unsqueeze(0) < plens.unsqueeze(1)).to(torch.long)
        # Distinct uids for weak vs strong so FSDP multi_modal_inputs cache is not reused with wrong pixels.
        weak_uids = [f"{u}__vision_sd_rlsd_weak" for u in valid_uids]

        def run_log_probs(
            prompts: list[str],
            mm_b: list[Optional[dict[str, Any]]],
            uids_list: list[Any],
        ) -> torch.Tensor:
            proto = self._build_prompt_batch(prompts, multi_modal_batch=mm_b, uids=uids_list)
            proto.batch["responses"] = responses
            proto.batch["response_mask"] = response_mask
            proto, pad_size = pad_dataproto_to_divisor(proto, self.actor_rollout_ref_wg.world_size)
            out = self.actor_rollout_ref_wg.compute_log_probs(proto)
            return unpad_dataproto(out, pad_size=pad_size).batch["old_log_probs"].cpu()

        log_strong = run_log_probs(strong_prompts, mm_strong, valid_uids)
        log_weak = run_log_probs(weak_prompts, mm_weak, weak_uids)
        return log_strong, log_weak, response_mask

    def _generate_text(self, prompts: list[str]) -> list[str]:
        gen_batch = self._build_prompt_batch(prompts)
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        try:
            gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
            output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            output = unpad_dataproto(output, pad_size=pad_size)
        finally:
            self.actor_rollout_ref_wg.release_rollout_engine()
        responses = output.batch["responses"]
        return self.tokenizer.batch_decode(
            responses, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _neutral_teacher_a_item(self, spans: list[str]) -> dict[str, Any]:
        return {
            "spans": spans,
            "calibrated_contribs": [0.0] * len(spans),
            "quality": self.reasoning_gate_delta,
            "quality_raw": self.reasoning_gate_delta,
            "support_quality": self.reasoning_gate_delta,
            "support_quality_raw": self.reasoning_gate_delta,
            "evidence_quality": self.reasoning_gate_delta,
            "evidence_quality_raw": self.reasoning_gate_delta,
            "full_score": self.reasoning_gate_delta,
        }

    def _init_teacher_a_engine(self) -> bool:
        if self.teacher_a_backend == "actor":
            # Actor backend does not need a standalone teacher engine.
            return True
        if self.teacher_a_disabled:
            return False
        if self._teacher_a_llm is not None:
            return True
        if self._teacher_a_init_error is not None:
            if self.teacher_a_fallback_on_error:
                return False
            raise RuntimeError(self._teacher_a_init_error)

        from vllm import LLM, SamplingParams

        tensor_parallel_size = max(1, int(os.getenv("VISION_SD_TEACHER_A_TP", "1")))
        gpu_memory_utilization = _float_env("VISION_SD_TEACHER_A_GPU_UTIL", 0.2)
        max_model_len = max(4096, int(os.getenv("VISION_SD_TEACHER_A_MAX_MODEL_LEN", "8192")))
        enforce_eager = bool(int(os.getenv("VISION_SD_TEACHER_A_ENFORCE_EAGER", "0")))
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
        teacher_ctx = (
            f"model={self.teacher_a_model_path}, tp={tensor_parallel_size}, "
            f"gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}, "
            f"enforce_eager={int(enforce_eager)}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}"
        )

        try:
            self._teacher_a_tokenizer = AutoTokenizer.from_pretrained(
                self.teacher_a_model_path,
                trust_remote_code=True,
                use_fast=True,
            )
            self._teacher_a_llm = LLM(
                model=self.teacher_a_model_path,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
            )
            self._teacher_a_sampling_params = SamplingParams(
                temperature=self.teacher_a_temperature,
                top_p=0.95,
                max_tokens=self.teacher_a_max_tokens,
            )
            return True
        except Exception as exc:
            message = (
                "Teacher A vLLM initialization failed. "
                f"{teacher_ctx}. "
                "This often happens when the Ray Runner actor has no visible GPU, "
                "or when the teacher model is too large for the reserved GPUs."
            )
            self._teacher_a_init_error = f"{message} Root cause: {exc}"
            if self.teacher_a_fallback_on_error:
                print(f"[vision_sd] WARNING: {self._teacher_a_init_error}")
                return False
            raise RuntimeError(self._teacher_a_init_error) from exc

    def _teacher_a_generate(self, prompts: list[str]) -> list[str]:
        if self.teacher_a_backend == "actor":
            return self._generate_text(prompts)
        if not self._init_teacher_a_engine():
            raise RuntimeError("Teacher A engine is unavailable for generation.")
        formatted_prompts = [
            self._teacher_a_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
        outputs = self._teacher_a_llm.generate(formatted_prompts, self._teacher_a_sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    def _scalarize_teacher_a_score(self, parsed: dict[str, Any]) -> float:
        support_score = clamp01(parsed.get("support_score", 0.0), default=0.0)
        reasoning_supported = 1.0 if _to_bool(parsed.get("reasoning_supported", False)) else 0.0
        answer_still_justified = 1.0 if _to_bool(parsed.get("answer_still_justified", False)) else 0.0
        missing_critical_evidence = 1.0 if _to_bool(parsed.get("missing_critical_evidence", False)) else 0.0

        score = (
            self.teacher_a_support_coef * support_score
            + self.teacher_a_reasoning_coef * reasoning_supported
            + self.teacher_a_answer_coef * answer_still_justified
            - self.teacher_a_missing_coef * missing_critical_evidence
        )
        return max(0.0, score)

    def _teacher_a_prompt(
        self,
        template_idx: int,
        question: str,
        perception_units: list[str],
        reasoning: str,
        answer: str,
    ) -> str:
        template = TEACHER_A_SUPPORT_PROMPTS[template_idx % len(TEACHER_A_SUPPORT_PROMPTS)]
        joined_perception = "\n".join(f"[{idx + 1}] {unit}" for idx, unit in enumerate(perception_units)) or "[1] "
        return template.format(
            question=question.strip(),
            perception=joined_perception,
            reasoning=reasoning.strip(),
            answer=answer.strip(),
        )

    def _teacher_a_qp_quality_prompt(self, template_idx: int, question: str, perception_units: list[str]) -> str:
        template = TEACHER_A_QP_QUALITY_PROMPTS[template_idx % len(TEACHER_A_QP_QUALITY_PROMPTS)]
        joined_perception = "\n".join(f"[{idx + 1}] {unit}" for idx, unit in enumerate(perception_units)) or "[1] "
        return template.format(question=question.strip(), perception=joined_perception)

    def _scalarize_teacher_a_qp_quality(self, parsed: dict[str, Any]) -> float:
        return clamp01(parsed.get("evidence_quality", 0.0), default=0.0)

    def _teacher_a_simple_ext_quality_prompt(self, template_idx: int, question: str, perception_units: list[str]) -> str:
        template = TEACHER_A_SIMPLE_EXT_QUALITY_PROMPTS[template_idx % len(TEACHER_A_SIMPLE_EXT_QUALITY_PROMPTS)]
        joined_perception = "\n".join(f"[{idx + 1}] {unit}" for idx, unit in enumerate(perception_units)) or "[1] "
        return template.format(question=question.strip(), perception=joined_perception)

    def _teacher_a_simple_solve_prompt(self, template_idx: int, question: str, perception_units: list[str]) -> str:
        template = TEACHER_A_SIMPLE_SOLVE_PROMPTS[template_idx % len(TEACHER_A_SIMPLE_SOLVE_PROMPTS)]
        joined_perception = "\n".join(f"[{idx + 1}] {unit}" for idx, unit in enumerate(perception_units)) or "[1] "
        return template.format(question=question.strip(), perception=joined_perception)

    def _scalarize_teacher_a_simple_ext_quality(self, parsed: dict[str, Any]) -> float:
        relevance = clamp01(parsed.get("relevance", 0.0), default=0.0)
        factuality = clamp01(parsed.get("factuality", 0.0), default=0.0)
        leakage_penalty = clamp01(parsed.get("leakage_penalty", 0.0), default=0.0)
        score = (
            self.simple_ext_rel_coef * relevance
            + self.simple_ext_fact_coef * factuality
            - self.simple_ext_leak_coef * leakage_penalty
        )
        return clamp01(score, default=0.0)

    def _scalarize_teacher_a_simple_solve(self, parsed: dict[str, Any]) -> float:
        return clamp01(parsed.get("solve_score", 0.0), default=0.0)

    def _select_utility_drop_indices(self, spans: list[str]) -> list[int]:
        if not spans:
            return []
        if len(spans) == 1:
            return [0]

        relation_keywords = {
            "left", "right", "above", "below", "between", "near", "behind", "front",
            "before", "after", "around", "inside", "outside", "over", "under",
        }
        ocr_keywords = {"text", "label", "sign", "ocr", "caption", "number", "digit", "reads"}
        count_keywords = {"total", "count", "many", "few", "amount", "sum", "difference", "ratio", "percent", "%"}

        def span_importance(span: str) -> tuple[int, int]:
            # Favor key factual evidence spans (numbers/relations/OCR/counting cues),
            # while avoiding low-content template-like fragments.
            span_lower = span.lower()
            token_count = len(span.split())
            alpha_chars = sum(ch.isalpha() for ch in span)
            has_number = bool(re.search(r"\d", span))
            has_relation = any(keyword in span_lower for keyword in relation_keywords)
            has_ocr = any(keyword in span_lower for keyword in ocr_keywords)
            has_count = any(keyword in span_lower for keyword in count_keywords)
            content_ok = token_count >= 3 and alpha_chars >= 8
            score = (
                3 * int(has_number)
                + 2 * int(has_relation)
                + 2 * int(has_ocr)
                + 1 * int(has_count)
                + 1 * int(content_ok)
            )
            return score, token_count

        ranked = []
        for idx, span in enumerate(spans):
            score, token_count = span_importance(span)
            ranked.append((score, token_count, idx))

        eligible = [item for item in ranked if item[0] > 0]
        pool = eligible if eligible else ranked

        if self.simple_utility_mode == "random_drop":
            drop_k = max(1, int(round(len(pool) * self.simple_utility_drop_ratio)))
            candidate_indices = [idx for _, _, idx in pool]
            perm = self._simple_rng.permutation(len(candidate_indices)).tolist()
            selected = [candidate_indices[i] for i in perm[:drop_k]]
            return sorted(selected)

        pool.sort(reverse=True)
        selected = [idx for _, _, idx in pool][: self.simple_utility_topk]
        if not selected:
            selected = [ranked[0][2]]
        return sorted(selected)

    def _compute_teacher_a_simple_signals(
        self,
        questions: list[str],
        perceptions: list[str],
    ) -> list[dict[str, float]]:
        if not self._init_teacher_a_engine():
            neutral = clamp01(self.simple_gate_delta, default=0.35)
            return [
                {
                    "q_ext": neutral,
                    "q_solve": 0.0,
                    "q_solve_signed": 0.0,
                    "solve_full": 0.0,
                    "solve_drop": 0.0,
                    "q_p": neutral,
                }
                for _ in perceptions
            ]

        outputs = [
            {
                "q_ext": 0.0,
                "q_solve": 0.0,
                "q_solve_signed": 0.0,
                "solve_full": 0.0,
                "solve_drop": 0.0,
                "q_p": 0.0,
            }
            for _ in perceptions
        ]

        valid_indices: list[int] = []
        valid_questions: list[str] = []
        valid_spans: list[list[str]] = []
        valid_reduced_spans: list[list[str]] = []

        for idx, (question, perception) in enumerate(zip(questions, perceptions)):
            spans = split_evidence_units(perception)[: self.max_perception_spans]
            if not spans:
                continue
            valid_indices.append(idx)
            valid_questions.append(question)
            valid_spans.append(spans)
            if self.simple_q_ext_only:
                continue
            drop_indices = set(self._select_utility_drop_indices(spans))
            reduced_spans = [
                (self.simple_utility_placeholder if span_idx in drop_indices else span)
                for span_idx, span in enumerate(spans)
            ]
            valid_reduced_spans.append(reduced_spans)

        if not valid_indices:
            return outputs

        # Batch teacher calls by stage to reduce per-sample RPC/scheduling overhead.
        ext_prompts = [
            self._teacher_a_simple_ext_quality_prompt(0, question, spans)
            for question, spans in zip(valid_questions, valid_spans)
        ]
        ext_raw_list = self._teacher_a_generate(ext_prompts)

        if self.simple_q_ext_only:
            for local_idx, batch_idx in enumerate(valid_indices):
                q_ext = self._scalarize_teacher_a_simple_ext_quality(
                    extract_json_object(ext_raw_list[local_idx])
                )
                outputs[batch_idx] = {
                    "q_ext": q_ext,
                    "q_solve": 0.0,
                    "q_solve_signed": 0.0,
                    "solve_full": 0.0,
                    "solve_drop": 0.0,
                    "q_p": q_ext,
                }
            return outputs

        solve_full_prompts = [
            self._teacher_a_simple_solve_prompt(0, question, spans)
            for question, spans in zip(valid_questions, valid_spans)
        ]
        solve_drop_prompts = [
            self._teacher_a_simple_solve_prompt(0, question, reduced_spans)
            for question, reduced_spans in zip(valid_questions, valid_reduced_spans)
        ]

        solve_full_raw_list = self._teacher_a_generate(solve_full_prompts)
        solve_drop_raw_list = self._teacher_a_generate(solve_drop_prompts)

        for local_idx, batch_idx in enumerate(valid_indices):
            q_ext = self._scalarize_teacher_a_simple_ext_quality(
                extract_json_object(ext_raw_list[local_idx])
            )
            solve_full = self._scalarize_teacher_a_simple_solve(
                extract_json_object(solve_full_raw_list[local_idx])
            )
            solve_drop = self._scalarize_teacher_a_simple_solve(
                extract_json_object(solve_drop_raw_list[local_idx])
            )
            q_solve_signed = float(solve_full - solve_drop)
            q_solve = clamp01(max(0.0, q_solve_signed), default=0.0)
            q_p = clamp01(
                self.simple_gate_lambda_ext * q_ext + self.simple_gate_lambda_solve * q_solve,
                default=0.0,
            )
            outputs[batch_idx] = {
                "q_ext": q_ext,
                "q_solve": q_solve,
                "q_solve_signed": q_solve_signed,
                "solve_full": float(solve_full),
                "solve_drop": float(solve_drop),
                "q_p": q_p,
            }

        return outputs

    def _calibrate_teacher_a_quality_component(self, key: str, value: float) -> float:
        value = clamp01(value, default=0.0)
        if not self.teacher_a_quality_calibration:
            return value

        stats = self._teacher_a_quality_stats[key]
        momentum = self.teacher_a_quality_calib_momentum

        if stats["mean"] is None:
            stats["mean"] = value
            stats["sq_mean"] = value * value
            return value

        stats["mean"] = momentum * float(stats["mean"]) + (1.0 - momentum) * value
        stats["sq_mean"] = momentum * float(stats["sq_mean"]) + (1.0 - momentum) * (value * value)

        variance = max(float(stats["sq_mean"]) - float(stats["mean"]) ** 2, self.teacher_a_quality_calib_eps)
        std = max(variance ** 0.5, self.teacher_a_quality_calib_eps)
        z = (value - float(stats["mean"])) / (std * self.teacher_a_quality_calib_temp)
        return float(1.0 / (1.0 + np.exp(-z)))

    def _compute_teacher_a(
        self,
        questions: list[str],
        perceptions: list[str],
        reasoning_blocks: list[str],
        answers: list[str],
    ):
        if not self._init_teacher_a_engine():
            return [
                self._neutral_teacher_a_item(split_evidence_units(perception)[: self.max_perception_spans])
                for perception in perceptions
            ]

        teacher_a = []
        for question, perception, reasoning_block, answer in zip(questions, perceptions, reasoning_blocks, answers):
            spans = split_evidence_units(perception)[: self.max_perception_spans]
            if not spans:
                teacher_a.append(
                    {
                        "spans": [],
                        "calibrated_contribs": [],
                        "quality": 0.0,
                        "quality_raw": 0.0,
                        "support_quality": 0.0,
                        "support_quality_raw": 0.0,
                        "evidence_quality": 0.0,
                        "evidence_quality_raw": 0.0,
                        "full_score": 0.0,
                    }
                )
                continue

            per_view_contribs: list[list[float]] = []
            full_scores: list[float] = []
            qp_quality_scores: list[float] = []
            for view_idx in range(self.teacher_a_num_views):
                if self.teacher_a_use_qp_quality:
                    qp_prompt = self._teacher_a_qp_quality_prompt(view_idx, question, spans)
                    qp_output = self._teacher_a_generate([qp_prompt])[0]
                    qp_quality_scores.append(self._scalarize_teacher_a_qp_quality(extract_json_object(qp_output)))
                prompts = [self._teacher_a_prompt(view_idx, question, spans, reasoning_block, answer)]
                for span_idx in range(len(spans)):
                    reduced_spans = [span for idx, span in enumerate(spans) if idx != span_idx]
                    prompts.append(self._teacher_a_prompt(view_idx, question, reduced_spans, reasoning_block, answer))

                raw_outputs = self._teacher_a_generate(prompts)
                scalar_scores = [self._scalarize_teacher_a_score(extract_json_object(output)) for output in raw_outputs]
                full_score = scalar_scores[0] if scalar_scores else 0.0
                full_scores.append(full_score)
                per_view_contribs.append([full_score - reduced_score for reduced_score in scalar_scores[1:]])

            contrib_matrix = np.array(per_view_contribs, dtype=np.float32) if per_view_contribs else np.zeros((1, len(spans)))
            mean_contrib = contrib_matrix.mean(axis=0) if contrib_matrix.size else np.zeros((len(spans),), dtype=np.float32)
            var_contrib = contrib_matrix.var(axis=0) if contrib_matrix.size else np.zeros((len(spans),), dtype=np.float32)
            confidence = np.exp(-self.teacher_a_conf_tau * var_contrib)
            calibrated = (confidence * np.maximum(0.0, mean_contrib)).tolist()
            support_quality_raw = float(np.mean(normalize_nonnegative(calibrated))) if calibrated else 0.0
            evidence_quality_raw = float(np.mean(qp_quality_scores)) if qp_quality_scores else support_quality_raw

            support_quality = self._calibrate_teacher_a_quality_component("support", support_quality_raw)
            evidence_quality = self._calibrate_teacher_a_quality_component("evidence", evidence_quality_raw)
            quality_raw = (
                self.teacher_a_quality_blend * evidence_quality_raw
                + (1.0 - self.teacher_a_quality_blend) * support_quality_raw
            )
            quality = (
                self.teacher_a_quality_blend * evidence_quality
                + (1.0 - self.teacher_a_quality_blend) * support_quality
            )

            teacher_a.append(
                {
                    "spans": spans,
                    "calibrated_contribs": calibrated,
                    "quality": quality,
                    "quality_raw": quality_raw,
                    "support_quality": support_quality,
                    "support_quality_raw": support_quality_raw,
                    "evidence_quality": evidence_quality,
                    "evidence_quality_raw": evidence_quality_raw,
                    "full_score": float(np.mean(full_scores)) if full_scores else 0.0,
                }
            )
        return teacher_a

    def _build_response_ids_from_1d_rows(
        self, id_rows: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad rollout token id rows to a batch; pad positions are masked out."""
        if not id_rows:
            return (
                torch.zeros(0, 0, dtype=torch.long),
                torch.zeros(0, 0, dtype=torch.long),
            )
        max_l = max(int(x.shape[0]) for x in id_rows)
        b = len(id_rows)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", None) or 0)
        responses = torch.full((b, max_l), pad_id, dtype=torch.long)
        response_mask = torch.zeros((b, max_l), dtype=torch.long)
        for i, row in enumerate(id_rows):
            L = int(row.shape[0])
            row_cpu = row.long().view(-1)
            responses[i, :L] = row_cpu[:L]
            response_mask[i, :L] = 1
        return responses, response_mask

    def _compute_teacher_b(
        self,
        questions: list[str],
        perceptions: list[str],
        reasoning_blocks: list[str],
        multi_modal_batch: list[Optional[dict[str, Any]]],
        uids: list[Any],
        student_reasoning_log_probs: list[torch.Tensor],
        teacher_b_prompts: Optional[list[str]] = None,
        teacher_b_response_texts: Optional[list[str]] = None,
        # When set, use exact rollout token ids (must match len(student LPs) per row) — avoids
        # decode→re-encode off-by-one vs student old_log_probs for GT-answer privilege mode.
        teacher_b_response_id_rows: Optional[list[Optional[torch.Tensor]]] = None,
    ):
        if teacher_b_prompts is None:
            with_p_prompts = [
                TEACHER_B_WITH_P_PROMPT.format(question=question, perception=perception)
                for question, perception in zip(questions, perceptions)
            ]
        else:
            with_p_prompts = list(teacher_b_prompts)
        if teacher_b_response_texts is None:
            response_texts = list(reasoning_blocks)
        else:
            response_texts = list(teacher_b_response_texts)
        n = len(questions)
        with_p_batch = self._build_prompt_batch(with_p_prompts, multi_modal_batch=multi_modal_batch, uids=uids)
        if teacher_b_response_id_rows is not None and len(teacher_b_response_id_rows) == n:
            id_rows: list[torch.Tensor] = []
            for i in range(n):
                row_opt = teacher_b_response_id_rows[i]
                if row_opt is not None:
                    id_rows.append(row_opt.detach().cpu().view(-1).long())
                else:
                    enc = self.tokenizer.encode(
                        response_texts[i],
                        add_special_tokens=False,
                        max_length=self.config.data.max_response_length,
                        truncation=True,
                    )
                    id_rows.append(torch.tensor(enc, dtype=torch.long))
            responses, response_mask = self._build_response_ids_from_1d_rows(id_rows)
        else:
            responses, response_mask = self._build_response_tensors(response_texts)

        with_p_batch.batch["responses"] = responses
        with_p_batch.batch["response_mask"] = response_mask

        with_p_batch, pad_size = pad_dataproto_to_divisor(with_p_batch, self.actor_rollout_ref_wg.world_size)
        with_p_log_probs = self.actor_rollout_ref_wg.compute_log_probs(with_p_batch)
        with_p_log_probs = unpad_dataproto(with_p_log_probs, pad_size=pad_size).batch["old_log_probs"]
        student_log_probs = torch.nn.utils.rnn.pad_sequence(
            student_reasoning_log_probs,
            batch_first=True,
            padding_value=0.0,
        ).to(with_p_log_probs.dtype)
        diff = (with_p_log_probs - student_log_probs) * response_mask

        token_credits = []
        deltas = []
        raw_diffs: list[torch.Tensor] = []
        for idx in range(diff.shape[0]):
            cur_mask = response_mask[idx].bool()
            cur_diff = diff[idx][cur_mask]
            if cur_diff.numel() == 0:
                token_credits.append(torch.zeros((0,), dtype=torch.float32))
                deltas.append(torch.tensor(0.0))
                raw_diffs.append(torch.zeros(0, dtype=torch.float32))
                continue

            raw_diffs.append(cur_diff.detach().float().cpu())
            mu = cur_diff.mean()
            sigma = cur_diff.std()
            normalized = (cur_diff - mu) / (sigma + 1e-6)
            positive_credit = torch.clamp(normalized, min=0.0)
            token_credits.append(positive_credit.cpu())
            deltas.append(cur_diff.mean().cpu())

        return deltas, token_credits, raw_diffs

    def _build_simple_rlsd_teacher_b_tail_lps_and_ids(
        self,
        batch: DataProto,
        valid_indices: list[int],
        parsed_samples: list[Optional[dict[str, str]]],
        response_ids: torch.Tensor,
        response_lengths: torch.Tensor,
        student_log_probs: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Exact rollout token ids and matching student LPs for the same Teacher B span (GT: post-perception tail; else reasoning only)."""
        tail_lps: list[torch.Tensor] = []
        tail_rids: list[torch.Tensor] = []
        for batch_idx in valid_indices:
            parsed = parsed_samples[batch_idx]
            assert parsed is not None
            cl = int(response_lengths[batch_idx].item())
            pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            pl = min(pl, cl)
            if self.teacher_b_gt_privilege and "ground_truth" in batch.non_tensor_batch:
                gt = str(batch.non_tensor_batch["ground_truth"][batch_idx] or "").strip()
                if gt:
                    tail = response_ids[batch_idx, pl:cl]
                    lps = student_log_probs[batch_idx, pl:cl]
                else:
                    reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
                    reasoning_len = min(reasoning_len, max(0, cl - pl))
                    tail = response_ids[batch_idx, pl : pl + reasoning_len]
                    lps = student_log_probs[batch_idx, pl : pl + reasoning_len]
            else:
                reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
                reasoning_len = min(reasoning_len, max(0, cl - pl))
                tail = response_ids[batch_idx, pl : pl + reasoning_len]
                lps = student_log_probs[batch_idx, pl : pl + reasoning_len]
            tail_lps.append(lps.clone().cpu())
            tail_rids.append(tail.clone().cpu())
        return tail_lps, tail_rids

    def _compute_teacher_b_delta_rollout_vs_qi(
        self,
        questions: list[str],
        multi_modal_batch: list[Optional[dict[str, Any]]],
        uids: list[Any],
        student_tail_log_probs: list[torch.Tensor],
        tail_id_rows: list[torch.Tensor],
    ):
        """``Δ = log p_rollout(y) - log p_{q,I}(y)`` on the same tail ids (stop-grad q+I jinja, no with_p forward)."""
        n = len(questions)
        if n != len(multi_modal_batch) or n != len(uids) or n != len(student_tail_log_probs) or n != len(tail_id_rows):
            raise ValueError("rollout_vs_qi: list length mismatch")
        qi_prompts = [self._render_scoring_prompt_qi_only(q, mm) for q, mm in zip(questions, multi_modal_batch)]
        uid_qi = [f"{u}__vision_sd_teach_b_qi" for u in uids]
        id_rows = [r.detach().cpu().view(-1).long() for r in tail_id_rows]
        responses, response_mask = self._build_response_ids_from_1d_rows(id_rows)
        qi_batch = self._build_prompt_batch(qi_prompts, multi_modal_batch=multi_modal_batch, uids=uid_qi)
        qi_batch.batch["responses"] = responses
        qi_batch.batch["response_mask"] = response_mask
        qi_batch, pad_size = pad_dataproto_to_divisor(qi_batch, self.actor_rollout_ref_wg.world_size)
        qi_out = self.actor_rollout_ref_wg.compute_log_probs(qi_batch)
        log_qi = unpad_dataproto(qi_out, pad_size=pad_size).batch["old_log_probs"].cpu().float()
        st_pad = torch.nn.utils.rnn.pad_sequence(
            [x.view(-1).float() for x in student_tail_log_probs],
            batch_first=True,
            padding_value=0.0,
        )
        rmaskf = response_mask.to(dtype=torch.float32)
        diff = (st_pad - log_qi) * rmaskf
        token_credits: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        raw_diffs: list[torch.Tensor] = []
        for idx in range(int(diff.shape[0])):
            cur_mask = response_mask[idx].bool()
            cur_diff = diff[idx][cur_mask]
            if cur_diff.numel() == 0:
                token_credits.append(torch.zeros((0,), dtype=torch.float32))
                deltas.append(torch.tensor(0.0))
                raw_diffs.append(torch.zeros(0, dtype=torch.float32))
                continue
            raw_diffs.append(cur_diff.detach().float().cpu())
            mu = cur_diff.mean()
            sigma = cur_diff.std()
            normalized = (cur_diff - mu) / (sigma + 1e-6)
            positive_credit = torch.clamp(normalized, min=0.0)
            token_credits.append(positive_credit.cpu())
            deltas.append(cur_diff.mean().cpu())
        return deltas, token_credits, raw_diffs

    def _perception_block_weights(self, perception_block: str, teacher_a_item: dict[str, Any]) -> list[float]:
        perception_text = re.search(r"<perception>([\s\S]*?)</perception>", perception_block).group(1).strip()
        spans = split_evidence_units(perception_text)[: len(teacher_a_item["calibrated_contribs"])]
        normalized = normalize_nonnegative(teacher_a_item["calibrated_contribs"])

        open_tag_len = len(self.tokenizer.encode("<perception>", add_special_tokens=False))
        close_tag_len = len(self.tokenizer.encode("</perception>", add_special_tokens=False))
        weights = [self.teacher_a_floor] * open_tag_len
        for span, span_credit in zip(spans, normalized):
            span_len = max(1, len(self.tokenizer.encode(span, add_special_tokens=False)))
            span_weight = self.teacher_a_floor + (1.0 - self.teacher_a_floor) * span_credit
            weights.extend([span_weight] * span_len)
        weights.extend([self.teacher_a_floor] * close_tag_len)

        block_len = max(1, len(self.tokenizer.encode(perception_block, add_special_tokens=False)))
        if len(weights) < block_len:
            weights.extend([self.teacher_a_floor] * (block_len - len(weights)))
        return weights[:block_len]

    def _response_weight_vector(
        self,
        response_text: str,
        response_length: int,
        teacher_a_item: Optional[dict[str, Any]],
        teacher_b_item: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        weights = torch.ones(response_length, dtype=torch.float32)
        parsed = parse_structured_response(response_text)
        if parsed is None:
            return weights

        perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
        reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))

        perception_len = min(perception_len, response_length)
        reasoning_len = min(reasoning_len, max(0, response_length - perception_len))

        if teacher_a_item is not None:
            p_weights = self._perception_block_weights(parsed["perception_block"], teacher_a_item)
            p_tensor = torch.tensor(p_weights[:perception_len], dtype=torch.float32)
            if len(p_tensor) < perception_len:
                p_tensor = torch.cat([p_tensor, torch.full((perception_len - len(p_tensor),), self.teacher_a_floor)])
            weights[:perception_len] = p_tensor[:perception_len]

        teacher_b_enabled = bool(teacher_b_item.get("enabled", True)) if teacher_b_item is not None else False
        if teacher_b_item is not None and teacher_b_enabled and reasoning_len > 0:
            r_tensor = teacher_b_item["weights"][:reasoning_len]
            if len(r_tensor) < reasoning_len:
                pad = torch.full((reasoning_len - len(r_tensor),), float(teacher_b_item["gate"]))
                r_tensor = torch.cat([r_tensor, pad])
            weights[perception_len : perception_len + reasoning_len] = r_tensor[:reasoning_len]

        return weights

    def _compute_credit_assignment(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, float]]:
        response_ids = batch.batch["responses"]
        response_lengths = torch.sum(batch.batch["response_mask"], dim=-1)
        student_log_probs = batch.batch["old_log_probs"].cpu()
        response_texts = []
        parsed_samples = []
        reasoning_student_log_probs = []
        for i in range(len(batch)):
            cur_len = int(response_lengths[i].item())
            text = self.tokenizer.decode(
                response_ids[i][:cur_len], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_texts.append(text)
            parsed = parse_structured_response(text)
            parsed_samples.append(parsed)

            if parsed is None:
                reasoning_student_log_probs.append(None)
                continue

            perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            perception_len = min(perception_len, cur_len)
            reasoning_len = min(reasoning_len, max(0, cur_len - perception_len))
            reasoning_student_log_probs.append(student_log_probs[i, perception_len : perception_len + reasoning_len])

        questions = batch.non_tensor_batch["question"].tolist()
        multi_modal_batch = batch.non_tensor_batch.get("multi_modal_data")
        if multi_modal_batch is None:
            multi_modal_batch = np.array([None] * len(batch), dtype=object)

        valid_indices = [idx for idx, parsed in enumerate(parsed_samples) if parsed is not None]
        teacher_a_map = {}
        teacher_b_map = {}
        effective_reject_delta = max(
            0.0,
            min(self.reasoning_gate_reject_delta, self.reasoning_gate_delta - self.reasoning_gate_margin),
        )

        if valid_indices:
            valid_questions = [questions[idx] for idx in valid_indices]
            valid_perceptions = [parsed_samples[idx]["perception_text"] for idx in valid_indices]
            valid_reasoning_blocks = [parsed_samples[idx]["reasoning_block"] for idx in valid_indices]
            valid_answers = [parsed_samples[idx]["answer_block"] for idx in valid_indices]
            valid_multi_modal = [multi_modal_batch[idx] for idx in valid_indices]
            valid_uids = [batch.non_tensor_batch["uid"][idx] for idx in valid_indices]
            valid_student_reasoning_log_probs = [reasoning_student_log_probs[idx] for idx in valid_indices]

            teacher_a_outputs = self._compute_teacher_a(
                valid_questions,
                valid_perceptions,
                valid_reasoning_blocks,
                valid_answers,
            )
            delta_b, teacher_b_credits, _raw_diffs_b = self._compute_teacher_b(
                valid_questions,
                valid_perceptions,
                valid_reasoning_blocks,
                valid_multi_modal,
                valid_uids,
                valid_student_reasoning_log_probs,
            )

            for local_idx, batch_idx in enumerate(valid_indices):
                teacher_a_item = teacher_a_outputs[local_idx]
                q_score = teacher_a_item["quality"]
                teacher_b_enabled = q_score >= effective_reject_delta
                if teacher_b_enabled:
                    gate = float(
                        torch.sigmoid(torch.tensor(self.reasoning_gate_beta * (q_score - self.reasoning_gate_delta))).item()
                    )
                    normalized_reasoning = normalize_nonnegative(teacher_b_credits[local_idx].tolist())
                    reasoning_weights = [
                        self.teacher_b_floor + (1.0 - self.teacher_b_floor) * gate * credit
                        for credit in normalized_reasoning
                    ]
                else:
                    gate = 0.0
                    reasoning_weights = []
                teacher_a_map[batch_idx] = teacher_a_item
                teacher_b_map[batch_idx] = {
                    "delta": float(delta_b[local_idx].item()),
                    "gate": gate,
                    "enabled": teacher_b_enabled,
                    "zone": (
                        "reject"
                        if q_score < effective_reject_delta
                        else "soft"
                        if q_score < self.reasoning_gate_delta
                        else "high"
                    ),
                    "weights": torch.tensor(reasoning_weights, dtype=torch.float32),
                }

        weight_tensor = torch.ones_like(batch.batch["response_mask"], dtype=torch.float32)
        teacher_a_scores, teacher_a_scores_raw = [], []
        teacher_a_support_scores, teacher_a_support_scores_raw = [], []
        teacher_a_evidence_scores, teacher_a_evidence_scores_raw = [], []
        teacher_b_deltas, gates = [], []
        teacher_b_rejected = 0
        teacher_b_soft = 0
        teacher_b_high = 0
        for idx, response_text in enumerate(response_texts):
            cur_len = int(response_lengths[idx].item())
            weights = self._response_weight_vector(
                response_text=response_text,
                response_length=cur_len,
                teacher_a_item=teacher_a_map.get(idx),
                teacher_b_item=teacher_b_map.get(idx),
            )
            weight_tensor[idx, :cur_len] = weights
            if idx in teacher_a_map:
                teacher_a_scores.append(teacher_a_map[idx]["quality"])
                teacher_a_scores_raw.append(teacher_a_map[idx].get("quality_raw", teacher_a_map[idx]["quality"]))
                teacher_a_support_scores.append(teacher_a_map[idx].get("support_quality", 0.0))
                teacher_a_support_scores_raw.append(teacher_a_map[idx].get("support_quality_raw", 0.0))
                teacher_a_evidence_scores.append(teacher_a_map[idx].get("evidence_quality", 0.0))
                teacher_a_evidence_scores_raw.append(teacher_a_map[idx].get("evidence_quality_raw", 0.0))
            if idx in teacher_b_map:
                teacher_b_deltas.append(teacher_b_map[idx]["delta"])
                gates.append(teacher_b_map[idx]["gate"])
                zone = teacher_b_map[idx].get("zone", "high")
                if zone == "reject":
                    teacher_b_rejected += 1
                elif zone == "soft":
                    teacher_b_soft += 1
                else:
                    teacher_b_high += 1

        metrics = {
            "credit/teacher_a_quality": float(np.mean(teacher_a_scores)) if teacher_a_scores else 0.0,
            "credit/teacher_a_quality_raw": float(np.mean(teacher_a_scores_raw)) if teacher_a_scores_raw else 0.0,
            "credit/teacher_a_support_quality": float(np.mean(teacher_a_support_scores)) if teacher_a_support_scores else 0.0,
            "credit/teacher_a_support_quality_raw": float(np.mean(teacher_a_support_scores_raw)) if teacher_a_support_scores_raw else 0.0,
            "credit/teacher_a_evidence_quality": float(np.mean(teacher_a_evidence_scores)) if teacher_a_evidence_scores else 0.0,
            "credit/teacher_a_evidence_quality_raw": float(np.mean(teacher_a_evidence_scores_raw)) if teacher_a_evidence_scores_raw else 0.0,
            "credit/teacher_b_delta": float(np.mean(teacher_b_deltas)) if teacher_b_deltas else 0.0,
            "credit/teacher_b_gate": float(np.mean(gates)) if gates else 0.0,
            "credit/teacher_b_reject_ratio": (float(teacher_b_rejected) / float(len(teacher_b_deltas))) if teacher_b_deltas else 0.0,
            "credit/teacher_b_soft_ratio": (float(teacher_b_soft) / float(len(teacher_b_deltas))) if teacher_b_deltas else 0.0,
            "credit/teacher_b_high_ratio": (float(teacher_b_high) / float(len(teacher_b_deltas))) if teacher_b_deltas else 0.0,
            "credit/reasoning_gate_effective_reject_delta": float(effective_reject_delta),
            "credit/reasoning_gate_delta": float(self.reasoning_gate_delta),
        }
        return weight_tensor * batch.batch["response_mask"], metrics

    def _compute_weighted_advantage(self, batch: DataProto, reward_metrics_raw: dict[str, list[float]]) -> dict[str, float]:
        if self.credit_mode in {"pure_grpo", "plain_grpo", "grpo_baseline"}:
            if self.config.algorithm.adv_estimator != "grpo":
                raise NotImplementedError("CreditDistillTrainer(pure_grpo) currently supports grpo only.")

            scores = torch.tensor(reward_metrics_raw["overall"], dtype=torch.float32)
            base_advantage = grpo_advantage_for_scores(scores, batch.batch["response_mask"], batch.non_tensor_batch["uid"])
            batch.batch["advantages"] = base_advantage
            batch.batch["returns"] = base_advantage
            return {
                "credit/mode_pure_grpo": 1.0,
                "credit/effective_weight_mean": 1.0,
                "credit/neg_adv_alpha": 0.0,
            }

        if self.credit_mode in {"simple_3s1g", "simple"}:
            return self._compute_weighted_advantage_simple_3s1g(batch, reward_metrics_raw)

        if self.credit_mode in {"simple_rlsd", "rlsd_perception", "rlsd_no_teacher_a"}:
            return self._compute_weighted_advantage_simple_rlsd(batch, reward_metrics_raw)

        if self.config.algorithm.adv_estimator != "grpo":
            raise NotImplementedError("CreditDistillTrainer currently supports grpo only.")

        scores = torch.tensor(reward_metrics_raw["overall"], dtype=torch.float32)
        base_advantage = grpo_advantage_for_scores(scores, batch.batch["response_mask"], batch.non_tensor_batch["uid"])
        credit_weights, credit_metrics = self._compute_credit_assignment(batch)
        neg_alpha = self.reasoning_neg_adv_alpha
        neg_weights = 1.0 + neg_alpha * (credit_weights - 1.0)
        effective_weights = torch.where(base_advantage >= 0.0, credit_weights, neg_weights)
        weighted_advantage = base_advantage * effective_weights
        batch.batch["advantages"] = weighted_advantage
        batch.batch["returns"] = weighted_advantage
        credit_metrics["credit/neg_adv_alpha"] = float(neg_alpha)
        credit_metrics["credit/effective_weight_mean"] = float(effective_weights.mean().item())
        return credit_metrics

    def _compute_weighted_advantage_simple_3s1g(
        self, batch: DataProto, reward_metrics_raw: dict[str, list[float]]
    ) -> dict[str, float]:
        if self.config.algorithm.adv_estimator != "grpo":
            raise NotImplementedError("CreditDistillTrainer(simple_3s1g) currently supports grpo only.")

        scores = torch.tensor(reward_metrics_raw["overall"], dtype=torch.float32)
        base_advantage = grpo_advantage_for_scores(scores, batch.batch["response_mask"], batch.non_tensor_batch["uid"])
        response_mask = batch.batch["response_mask"]

        response_ids = batch.batch["responses"]
        response_lengths = torch.sum(response_mask, dim=-1)
        student_log_probs = batch.batch["old_log_probs"].cpu()
        response_texts = []
        parsed_samples = []
        reasoning_student_log_probs = []
        for i in range(len(batch)):
            cur_len = int(response_lengths[i].item())
            text = self.tokenizer.decode(
                response_ids[i][:cur_len], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_texts.append(text)
            parsed = parse_structured_response(text)
            parsed_samples.append(parsed)
            if parsed is None:
                reasoning_student_log_probs.append(None)
                continue

            perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            perception_len = min(perception_len, cur_len)
            reasoning_len = min(reasoning_len, max(0, cur_len - perception_len))
            reasoning_student_log_probs.append(student_log_probs[i, perception_len : perception_len + reasoning_len])

        questions = batch.non_tensor_batch["question"].tolist()
        multi_modal_batch = batch.non_tensor_batch.get("multi_modal_data")
        if multi_modal_batch is None:
            multi_modal_batch = np.array([None] * len(batch), dtype=object)

        valid_indices = [idx for idx, parsed in enumerate(parsed_samples) if parsed is not None]
        signal_map: dict[int, dict[str, float]] = {}
        teacher_b_map: dict[int, dict[str, Any]] = {}

        if valid_indices:
            valid_questions = [questions[idx] for idx in valid_indices]
            valid_perceptions = [parsed_samples[idx]["perception_text"] for idx in valid_indices]
            valid_reasoning_blocks = [parsed_samples[idx]["reasoning_block"] for idx in valid_indices]
            valid_multi_modal = [multi_modal_batch[idx] for idx in valid_indices]
            valid_uids = [batch.non_tensor_batch["uid"][idx] for idx in valid_indices]
            valid_student_reasoning_log_probs = [reasoning_student_log_probs[idx] for idx in valid_indices]

            signal_outputs = self._compute_teacher_a_simple_signals(valid_questions, valid_perceptions)
            delta_b, teacher_b_credits, _raw_diffs_b_3s = self._compute_teacher_b(
                valid_questions,
                valid_perceptions,
                valid_reasoning_blocks,
                valid_multi_modal,
                valid_uids,
                valid_student_reasoning_log_probs,
            )

            for local_idx, batch_idx in enumerate(valid_indices):
                sig = signal_outputs[local_idx]
                q_p = sig["q_p"]
                gate = float(torch.sigmoid(torch.tensor(self.simple_gate_beta * (q_p - self.simple_gate_delta))).item())
                support_credits = normalize_nonnegative(teacher_b_credits[local_idx].tolist())
                teacher_b_map[batch_idx] = {
                    "delta": float(delta_b[local_idx].item()),
                    "gate": gate,
                    "support": torch.tensor(support_credits, dtype=torch.float32),
                }
                signal_map[batch_idx] = sig

        raw_w = torch.ones_like(response_mask, dtype=torch.float32)
        q_ext_vals, q_solve_vals, q_solve_signed_vals, solve_full_vals, solve_drop_vals, q_p_vals, gate_vals, delta_vals = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx, response_text in enumerate(response_texts):
            cur_len = int(response_lengths[idx].item())
            parsed = parsed_samples[idx]
            if parsed is None or cur_len == 0:
                continue

            sig = signal_map.get(
                idx,
                {
                    "q_ext": self.simple_gate_delta,
                    "q_solve": 0.0,
                    "q_solve_signed": 0.0,
                    "solve_full": 0.0,
                    "solve_drop": 0.0,
                    "q_p": self.simple_gate_delta,
                },
            )
            q_ext, q_solve, q_p = sig["q_ext"], sig["q_solve"], sig["q_p"]
            q_ext_vals.append(q_ext)
            q_solve_vals.append(q_solve)
            q_solve_signed_vals.append(float(sig.get("q_solve_signed", 0.0)))
            solve_full_vals.append(float(sig.get("solve_full", 0.0)))
            solve_drop_vals.append(float(sig.get("solve_drop", 0.0)))
            q_p_vals.append(q_p)

            perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            perception_len = min(perception_len, cur_len)
            reasoning_len = min(reasoning_len, max(0, cur_len - perception_len))

            p_weight = self.simple_perception_floor + (1.0 - self.simple_perception_floor) * q_p
            raw_w[idx, :perception_len] = float(p_weight)

            b_item = teacher_b_map.get(idx)
            if b_item is not None and reasoning_len > 0:
                gate = float(b_item["gate"])
                support = b_item["support"][:reasoning_len]
                if support.numel() < reasoning_len:
                    support = torch.cat([support, torch.zeros((reasoning_len - support.numel(),), dtype=torch.float32)])
                centered = support - support.mean() if support.numel() > 0 else support
                raw_reasoning = 1.0 + gate * self.simple_reasoning_support_scale * centered
                raw_w[idx, perception_len : perception_len + reasoning_len] = raw_reasoning
                gate_vals.append(gate)
                delta_vals.append(float(b_item["delta"]))

        modulation = torch.ones_like(raw_w, dtype=torch.float32)
        valid_mask = response_mask.bool()
        for i in range(raw_w.shape[0]):
            cur_mask = valid_mask[i]
            if not bool(cur_mask.any()):
                continue
            cur_w = raw_w[i][cur_mask]
            cur_mean = cur_w.mean()
            clipped = torch.clamp(cur_w - cur_mean, min=-self.simple_mod_clip, max=self.simple_mod_clip)
            cur_mod = 1.0 + self.simple_mod_lambda * clipped
            cur_mod = torch.clamp(cur_mod, min=self.simple_mod_mult_min, max=self.simple_mod_mult_max)
            modulation[i][cur_mask] = cur_mod

        # Per-segment max token weights (structured parse only).
        _T3 = raw_w.shape[1]
        _perc_m3 = torch.zeros_like(response_mask, dtype=torch.float32)
        _reas_m3 = torch.zeros_like(response_mask, dtype=torch.float32)
        for idx, _ in enumerate(response_texts):
            cur_len = int(response_lengths[idx].item())
            parsed = parsed_samples[idx]
            if parsed is None or cur_len == 0:
                continue
            pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            rl = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            pl = min(pl, cur_len)
            rl = min(rl, max(0, cur_len - pl))
            if pl > 0:
                _perc_m3[idx, : min(pl, _T3)] = 1.0
            if rl > 0:
                s = min(pl, _T3)
                e = min(pl + rl, _T3)
                if s < e:
                    _reas_m3[idx, s:e] = 1.0
        _vm3 = response_mask.float()
        _peff3 = _perc_m3 * _vm3
        _reff3 = _reas_m3 * _vm3
        _p_d3 = float(_peff3.sum().item())
        _r_d3 = float(_reff3.sum().item())
        raw_w_max_perc3 = float((raw_w * _peff3).max().item()) if _p_d3 > 1e-8 else 0.0
        raw_w_max_reas3 = float((raw_w * _reff3).max().item()) if _r_d3 > 1e-8 else 0.0
        mod_max_perc3 = float((modulation * _peff3).max().item()) if _p_d3 > 1e-8 else 0.0
        mod_max_reas3 = float((modulation * _reff3).max().item()) if _r_d3 > 1e-8 else 0.0

        weighted_advantage = base_advantage * modulation * response_mask
        batch.batch["advantages"] = weighted_advantage
        batch.batch["returns"] = weighted_advantage

        return {
            "credit/mode_simple_3s1g": 1.0,
            "credit/simple_q_ext": float(np.mean(q_ext_vals)) if q_ext_vals else 0.0,
            "credit/simple_q_solve": float(np.mean(q_solve_vals)) if q_solve_vals else 0.0,
            "credit/simple_q_solve_signed": float(np.mean(q_solve_signed_vals)) if q_solve_signed_vals else 0.0,
            "credit/simple_q_solve_neg_ratio": (
                float(np.mean([1.0 if val < 0.0 else 0.0 for val in q_solve_signed_vals])) if q_solve_signed_vals else 0.0
            ),
            "credit/simple_solve_full": float(np.mean(solve_full_vals)) if solve_full_vals else 0.0,
            "credit/simple_solve_drop": float(np.mean(solve_drop_vals)) if solve_drop_vals else 0.0,
            "credit/simple_q_p": float(np.mean(q_p_vals)) if q_p_vals else 0.0,
            "credit/simple_gate": float(np.mean(gate_vals)) if gate_vals else 0.0,
            "credit/simple_teacher_b_delta": float(np.mean(delta_vals)) if delta_vals else 0.0,
            "credit/simple_raw_weight_mean": float((raw_w * response_mask).sum().item() / max(1.0, response_mask.sum().item())),
            "credit/simple_mod_mean": float((modulation * response_mask).sum().item() / max(1.0, response_mask.sum().item())),
            "credit/simple_mod_lambda": float(self.simple_mod_lambda),
            "credit/simple_mod_clip": float(self.simple_mod_clip),
            "credit/simple_raw_w_max_perception": raw_w_max_perc3,
            "credit/simple_raw_w_max_reasoning": raw_w_max_reas3,
            "credit/simple_mod_max_perception": mod_max_perc3,
            "credit/simple_mod_max_reasoning": mod_max_reas3,
        }

    def _compute_weighted_advantage_simple_rlsd(
        self, batch: DataProto, reward_metrics_raw: dict[str, list[float]]
    ) -> dict[str, float]:
        """Perception credit from Δ log p (same prompt, original vs masked image) on perception tokens; Teacher B unchanged; no Teacher A."""
        if self.config.algorithm.adv_estimator != "grpo":
            raise NotImplementedError("CreditDistillTrainer(simple_rlsd) currently supports grpo only.")

        scores = torch.tensor(reward_metrics_raw["overall"], dtype=torch.float32)
        response_mask = batch.batch["response_mask"]
        n_batch = len(batch)
        rlsd_delta_bonus = torch.zeros(n_batch, dtype=torch.float32)

        response_ids = batch.batch["responses"]
        response_lengths = torch.sum(response_mask, dim=-1)
        student_log_probs = batch.batch["old_log_probs"].cpu()
        response_texts = []
        parsed_samples = []
        reasoning_student_log_probs = []
        for i in range(len(batch)):
            cur_len = int(response_lengths[i].item())
            text = self.tokenizer.decode(
                response_ids[i][:cur_len], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_texts.append(text)
            parsed = parse_structured_response(text)
            parsed_samples.append(parsed)
            if parsed is None:
                reasoning_student_log_probs.append(None)
                continue

            perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            perception_len = min(perception_len, cur_len)
            reasoning_len = min(reasoning_len, max(0, cur_len - perception_len))
            reasoning_student_log_probs.append(student_log_probs[i, perception_len : perception_len + reasoning_len])

        questions = batch.non_tensor_batch["question"].tolist()
        multi_modal_batch = batch.non_tensor_batch.get("multi_modal_data")
        if multi_modal_batch is None:
            multi_modal_batch = np.array([None] * len(batch), dtype=object)

        valid_indices = [idx for idx, parsed in enumerate(parsed_samples) if parsed is not None]
        teacher_b_map: dict[int, dict[str, Any]] = {}
        log_ratio_mean_vals: list[float] = []
        w_rows: dict[int, torch.Tensor] = {}

        if valid_indices:
            valid_questions = [questions[idx] for idx in valid_indices]
            valid_perceptions = [parsed_samples[idx]["perception_text"] for idx in valid_indices]
            valid_reasoning_blocks = [parsed_samples[idx]["reasoning_block"] for idx in valid_indices]
            valid_multi_modal = [multi_modal_batch[idx] for idx in valid_indices]
            valid_uids = [batch.non_tensor_batch["uid"][idx] for idx in valid_indices]
            valid_student_reasoning_log_probs = [reasoning_student_log_probs[idx] for idx in valid_indices]

            log_strong, log_weak, p_response_mask = self._compute_perception_image_text_logprob_diff(
                valid_indices=valid_indices,
                questions=questions,
                response_ids=response_ids,
                response_lengths=response_lengths,
                parsed_samples=parsed_samples,
                multi_modal_batch=multi_modal_batch,
                valid_uids=valid_uids,
            )
            mask_f = p_response_mask.float()
            log_ratio = (log_strong - log_weak).detach()
            log_ratio = torch.clamp(
                log_ratio, min=-_RLSD_LOG_RATIO_EXP_SANITY, max=_RLSD_LOG_RATIO_EXP_SANITY
            )
            log_ratio_mean = (log_ratio * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)
            log_ratio_mean_vals = [float(log_ratio_mean[i].item()) for i in range(log_ratio_mean.shape[0])]
            for local_idx, batch_idx in enumerate(valid_indices):
                rlsd_delta_bonus[batch_idx] = max(0.0, log_ratio_mean_vals[local_idx])

            alloc = self.rlsd_perception_credit_alloc
            if alloc in {"softmax", "soft"}:
                neg_inf = torch.finfo(log_ratio.dtype).min
                logits = self.rlsd_softmax_beta * log_ratio
                logits = torch.where(mask_f > 0, logits, torch.full_like(logits, neg_inf))
                plen = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
                w = torch.softmax(logits, dim=-1) * plen
                w = w * mask_f
            elif alloc in {"proportional", "prop", "likelihood_ratio", "lratio"}:
                w = torch.exp(log_ratio)
                if self.rlsd_normalize_per_seq:
                    denom = (w * mask_f).sum(dim=-1, keepdim=True) / mask_f.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1.0)
                    w = w / denom.clamp(min=1e-6)
                if self.rlsd_lambda_warmup_steps > 0:
                    lam = min(1.0, float(self.global_step) / float(self.rlsd_lambda_warmup_steps))
                    w = (1.0 - lam) + lam * w
            else:
                w = torch.exp(log_ratio)
                if self.rlsd_normalize_per_seq:
                    denom = (w * mask_f).sum(dim=-1, keepdim=True) / mask_f.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1.0)
                    w = w / denom.clamp(min=1e-6)
                if self.rlsd_lambda_warmup_steps > 0:
                    lam = min(1.0, float(self.global_step) / float(self.rlsd_lambda_warmup_steps))
                    w = (1.0 - lam) + lam * w
                w = torch.clamp(w, min=self.rlsd_w_clip_low, max=self.rlsd_w_clip_high)

            if self.teacher_b_reasoning_delta_mode == "rollout_vs_qi":
                tail_lps, tail_rids = self._build_simple_rlsd_teacher_b_tail_lps_and_ids(
                    batch,
                    valid_indices,
                    parsed_samples,
                    response_ids,
                    response_lengths,
                    student_log_probs,
                )
                delta_b, teacher_b_credits, raw_b_diffs = self._compute_teacher_b_delta_rollout_vs_qi(
                    valid_questions,
                    valid_multi_modal,
                    valid_uids,
                    tail_lps,
                    tail_rids,
                )
            else:
                tb_pt: Optional[list[str]] = None
                tb_rt: Optional[list[str]] = None
                tb_lps: Optional[list[torch.Tensor]] = None
                tb_rids: Optional[list[Optional[torch.Tensor]]] = None
                if self.teacher_b_gt_privilege and "ground_truth" in batch.non_tensor_batch:
                    tb_pt = []
                    tb_rt = []
                    tb_lps = []
                    tb_rids = []
                    for vidx, batch_idx in enumerate(valid_indices):
                        parsed = parsed_samples[batch_idx]
                        cl = int(response_lengths[batch_idx].item())
                        pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
                        pl = min(pl, cl)
                        gt = str(batch.non_tensor_batch["ground_truth"][batch_idx] or "").strip()
                        if gt:
                            tb_pt.append(
                                TEACHER_B_WITH_P_GT_ANSWER_PROMPT.format(
                                    question=valid_questions[vidx],
                                    perception=valid_perceptions[vidx],
                                    ground_truth=gt,
                                )
                            )
                            # Exact rollout token ids; do not decode→re-encode (avoids 1266 vs 1267 mismatch with student LPs).
                            tail_ids = response_ids[batch_idx][pl:cl]
                            tb_rt.append(valid_reasoning_blocks[vidx])
                            tb_rids.append(tail_ids.detach().cpu())
                            tb_lps.append(student_log_probs[batch_idx, pl:cl].clone().cpu())
                        else:
                            tb_pt.append(
                                TEACHER_B_WITH_P_PROMPT.format(
                                    question=valid_questions[vidx], perception=valid_perceptions[vidx]
                                )
                            )
                            tb_rt.append(valid_reasoning_blocks[vidx])
                            reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
                            reasoning_len = min(reasoning_len, max(0, cl - pl))
                            tb_rids.append(
                                response_ids[batch_idx, pl : pl + reasoning_len].detach().cpu()
                            )
                            tb_lps.append(student_log_probs[batch_idx, pl : pl + reasoning_len].clone().cpu())

                if tb_rids is None:
                    tb_rids = []
                    for batch_idx in valid_indices:
                        parsed = parsed_samples[batch_idx]
                        assert parsed is not None
                        cl = int(response_lengths[batch_idx].item())
                        pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
                        pl = min(pl, cl)
                        reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
                        reasoning_len = min(reasoning_len, max(0, cl - pl))
                        tb_rids.append(
                            response_ids[batch_idx, pl : pl + reasoning_len].detach().cpu()
                        )

                student_b_lps = tb_lps if tb_lps is not None else valid_student_reasoning_log_probs
                delta_b, teacher_b_credits, raw_b_diffs = self._compute_teacher_b(
                    valid_questions,
                    valid_perceptions,
                    valid_reasoning_blocks,
                    valid_multi_modal,
                    valid_uids,
                    student_b_lps,
                    teacher_b_prompts=tb_pt,
                    teacher_b_response_texts=tb_rt,
                    teacher_b_response_id_rows=tb_rids,
                )
            _use_b_rlsvr = self.simple_teacher_b_mod in {"rlsvr", "rlvr", "rlvr_paper", "paper"}

            for local_idx, batch_idx in enumerate(valid_indices):
                w_rows[batch_idx] = w[local_idx].clone()
                lr_m = log_ratio_mean_vals[local_idx]
                lr_m_gate = lr_m + self.simple_rlsd_gate_lr_m_bias
                gate_pre = self.rlsd_gate_k * lr_m_gate - self.simple_gate_delta
                if self.simple_rlsd_teacher_b_gate_fixed is not None:
                    gate = float(np.clip(self.simple_rlsd_teacher_b_gate_fixed, 0.0, 1.0))
                else:
                    gate = float(torch.sigmoid(torch.tensor(self.simple_gate_beta * gate_pre)).item())
                support_credits = normalize_nonnegative(teacher_b_credits[local_idx].tolist())
                cl_e = int(response_lengths[batch_idx].item())
                p_e = parsed_samples[batch_idx]
                if p_e is not None:
                    pl_e = max(1, len(self.tokenizer.encode(p_e["perception_block"], add_special_tokens=False)))
                    pl_e = min(pl_e, cl_e)
                else:
                    pl_e = 0
                _sup_t = torch.tensor(support_credits, dtype=torch.float32)
                b_apply = int(min(int(_sup_t.shape[0]), max(0, cl_e - pl_e)))
                _entry: dict[str, Any] = {
                    "delta": float(delta_b[local_idx].item()),
                    "gate": gate,
                    "support": _sup_t,
                    "gate_pre_linear": float(gate_pre),
                    "b_apply_len": b_apply,
                }
                if _use_b_rlsvr:
                    _entry["raw_reasoning_diff"] = raw_b_diffs[local_idx]
                teacher_b_map[batch_idx] = _entry

        raw_w = torch.ones_like(response_mask, dtype=torch.float32)
        gate_vals: list[float] = []
        delta_vals: list[float] = []
        gate_pre_linear_vals: list[float] = []

        for idx, _ in enumerate(response_texts):
            cur_len = int(response_lengths[idx].item())
            parsed = parsed_samples[idx]
            if parsed is None or cur_len == 0:
                continue

            perception_len = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            reasoning_len = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            perception_len = min(perception_len, cur_len)
            reasoning_len = min(reasoning_len, max(0, cur_len - perception_len))

            row_w = w_rows.get(idx)
            if row_w is not None:
                plen = min(perception_len, row_w.numel())
                blend = self.simple_perception_floor + (1.0 - self.simple_perception_floor) * row_w[:plen]
                raw_w[idx, :plen] = blend.to(raw_w.dtype)

            b_item = teacher_b_map.get(idx)
            b_apply = int(b_item.get("b_apply_len", reasoning_len)) if b_item is not None else reasoning_len
            if b_item is not None and b_apply > 0:
                gate = float(b_item["gate"])
                if self.simple_teacher_b_mod in {"rlsvr", "rlvr", "rlvr_paper", "paper"}:
                    raw_w[idx, perception_len : perception_len + b_apply] = 1.0
                else:
                    support = b_item["support"][:b_apply]
                    if support.numel() < b_apply:
                        support = torch.cat([support, torch.zeros((b_apply - support.numel(),), dtype=torch.float32)])
                    centered = support - support.mean() if support.numel() > 0 else support
                    raw_reasoning = 1.0 + gate * self.simple_reasoning_support_scale * centered
                    raw_w[idx, perception_len : perception_len + b_apply] = raw_reasoning
                gate_vals.append(gate)
                delta_vals.append(float(b_item["delta"]))
                gate_pre_linear_vals.append(float(b_item.get("gate_pre_linear", 0.0)))

        scores = scores + self.rlsd_reward_delta_lambda * rlsd_delta_bonus.to(device=scores.device, dtype=scores.dtype)
        base_advantage = grpo_advantage_for_scores(scores, batch.batch["response_mask"], batch.non_tensor_batch["uid"])
        base_advantage = base_advantage.to(device=response_mask.device, dtype=torch.float32)

        # RLVR-style modulation on the reasoning span:
        # - default: apply after simple_mod by overwriting reasoning multipliers (old behavior).
        # - VISION_SD_RLSVR_BEFORE_SIMPLE_MOD=1: write rlsvr weights into raw_w first, then run one unified simple_mod.
        use_b_rlsvr = self.simple_teacher_b_mod in {"rlsvr", "rlvr", "rlvr_paper", "paper"}
        apply_rlsvr_before_simple_mod = bool(use_b_rlsvr and self.rlsvr_before_simple_mod)
        w_rlsvr_sum = 0.0
        w_rlsvr_n = 0
        if use_b_rlsvr:
            lo_w = 1.0 - float(self.teacher_b_paper_w_eps)
            hi_w = 1.0 + float(self.teacher_b_paper_w_eps)
            dcl = float(self.teacher_b_paper_delta_clip)
            sign_a = torch.sign(base_advantage[:, 0:1].to(torch.float32))
            for idx, _ in enumerate(response_texts):
                b_item = teacher_b_map.get(idx)
                if b_item is None or "raw_reasoning_diff" not in b_item:
                    continue
                rd = b_item["raw_reasoning_diff"]
                if not isinstance(rd, torch.Tensor) or rd.numel() == 0:
                    continue
                parsed = parsed_samples[idx]
                if parsed is None:
                    continue
                cur_len = int(response_lengths[idx].item())
                pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
                rl = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
                pl = min(pl, cur_len)
                rl = min(rl, max(0, cur_len - pl))
                tail_avail = max(0, cur_len - pl)
                b_apply = int(b_item.get("b_apply_len", rl))
                r_apply = int(min(b_apply, tail_avail, int(rd.numel())))
                if r_apply <= 0:
                    continue
                drow = rd[:r_apply].to(device=base_advantage.device, dtype=torch.float32)
                drow = torch.clamp(drow, -dcl, dcl)
                wp = torch.exp(sign_a[idx, 0] * drow).to(dtype=base_advantage.dtype)
                wp = torch.clamp(wp, min=lo_w, max=hi_w)
                L = int(wp.numel())
                if L > 0:
                    if apply_rlsvr_before_simple_mod:
                        raw_w[idx, pl : pl + L] = wp.to(dtype=raw_w.dtype)
                    else:
                        # defer to post-simple_mod overwrite path below
                        pass
                    w_rlsvr_sum += float(wp.sum().item())
                    w_rlsvr_n += L

        if self.simple_mod_bypass:
            modulation = raw_w.to(dtype=torch.float32)
        else:
            modulation = torch.ones_like(raw_w, dtype=torch.float32)
            valid_mask = response_mask.bool()
            for i in range(raw_w.shape[0]):
                cur_mask = valid_mask[i]
                if not bool(cur_mask.any()):
                    continue
                cur_w = raw_w[i][cur_mask]
                cur_mean = cur_w.mean()
                clipped = torch.clamp(cur_w - cur_mean, min=-self.simple_mod_clip, max=self.simple_mod_clip)
                cur_mod = 1.0 + self.simple_mod_lambda * clipped
                cur_mod = torch.clamp(cur_mod, min=self.simple_mod_mult_min, max=self.simple_mod_mult_max)
                modulation[i][cur_mask] = cur_mod

        eff_mod = modulation
        if use_b_rlsvr and (not apply_rlsvr_before_simple_mod):
            eff_mod = modulation.clone()
            lo_w = 1.0 - float(self.teacher_b_paper_w_eps)
            hi_w = 1.0 + float(self.teacher_b_paper_w_eps)
            dcl = float(self.teacher_b_paper_delta_clip)
            sign_a = torch.sign(base_advantage[:, 0:1].to(torch.float32))
            for idx, _ in enumerate(response_texts):
                b_item = teacher_b_map.get(idx)
                if b_item is None or "raw_reasoning_diff" not in b_item:
                    continue
                rd = b_item["raw_reasoning_diff"]
                if not isinstance(rd, torch.Tensor) or rd.numel() == 0:
                    continue
                parsed = parsed_samples[idx]
                if parsed is None:
                    continue
                cur_len = int(response_lengths[idx].item())
                pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
                rl = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
                pl = min(pl, cur_len)
                rl = min(rl, max(0, cur_len - pl))
                tail_avail = max(0, cur_len - pl)
                b_apply = int(b_item.get("b_apply_len", rl))
                r_apply = int(min(b_apply, tail_avail, int(rd.numel())))
                if r_apply <= 0:
                    continue
                drow = rd[:r_apply].to(device=base_advantage.device, dtype=torch.float32)
                drow = torch.clamp(drow, -dcl, dcl)
                wp = torch.exp(sign_a[idx, 0] * drow).to(dtype=base_advantage.dtype)
                wp = torch.clamp(wp, min=lo_w, max=hi_w)
                L = int(wp.numel())
                if L > 0:
                    eff_mod[idx, pl : pl + L] = wp

        # Per-segment means (structured parse only): perception [0,pl), reasoning [pl,pl+rl); tail tokens excluded.
        _T = raw_w.shape[1]
        _perc_m = torch.zeros_like(response_mask, dtype=torch.float32)
        _reas_m = torch.zeros_like(response_mask, dtype=torch.float32)
        for idx, _ in enumerate(response_texts):
            cur_len = int(response_lengths[idx].item())
            parsed = parsed_samples[idx]
            if parsed is None or cur_len == 0:
                continue
            pl = max(1, len(self.tokenizer.encode(parsed["perception_block"], add_special_tokens=False)))
            rl = max(1, len(self.tokenizer.encode(parsed["reasoning_block"], add_special_tokens=False)))
            pl = min(pl, cur_len)
            rl = min(rl, max(0, cur_len - pl))
            if pl > 0:
                _perc_m[idx, : min(pl, _T)] = 1.0
            if rl > 0:
                s = min(pl, _T)
                e = min(pl + rl, _T)
                if s < e:
                    _reas_m[idx, s:e] = 1.0
        _vm = response_mask.float()
        _peff = _perc_m * _vm
        _reff = _reas_m * _vm
        _p_den = float(_peff.sum().item())
        _r_den = float(_reff.sum().item())
        raw_w_mean_perc = float((raw_w * _peff).sum().item() / _p_den) if _p_den > 1e-8 else 0.0
        raw_w_mean_reas = float((raw_w * _reff).sum().item() / _r_den) if _r_den > 1e-8 else 0.0
        mod_mean_perc = float((modulation * _peff).sum().item() / _p_den) if _p_den > 1e-8 else 0.0
        mod_mean_reas = float((modulation * _reff).sum().item() / _r_den) if _r_den > 1e-8 else 0.0
        raw_w_max_perc = float((raw_w * _peff).max().item()) if _p_den > 1e-8 else 0.0
        raw_w_max_reas = float((raw_w * _reff).max().item()) if _r_den > 1e-8 else 0.0
        mod_max_perc = float((modulation * _peff).max().item()) if _p_den > 1e-8 else 0.0
        mod_max_reas = float((modulation * _reff).max().item()) if _r_den > 1e-8 else 0.0
        em_mean_perc = float((eff_mod * _peff).sum().item() / _p_den) if _p_den > 1e-8 else 0.0
        em_mean_reas = float((eff_mod * _reff).sum().item() / _r_den) if _r_den > 1e-8 else 0.0

        weighted_advantage = base_advantage * eff_mod * response_mask
        batch.batch["advantages"] = weighted_advantage
        batch.batch["returns"] = weighted_advantage

        return {
            "credit/mode_simple_rlsd": 1.0,
            "credit/simple_mod_bypass": 1.0 if self.simple_mod_bypass else 0.0,
            "credit/simple_teacher_b_mod_rlsvr": 1.0 if use_b_rlsvr else 0.0,
            "credit/rlsvr_before_simple_mod": 1.0 if apply_rlsvr_before_simple_mod else 0.0,
            "credit/teacher_b_rlsvr_w_mean": (w_rlsvr_sum / w_rlsvr_n) if w_rlsvr_n else 0.0,
            "credit/simple_mod_mean_reasoning_effective": em_mean_reas,
            "credit/simple_mod_mean_perception_effective": em_mean_perc,
            "credit/rlsd_perception_credit_softmax": 1.0
            if self.rlsd_perception_credit_alloc in {"softmax", "soft"}
            else 0.0,
            "credit/rlsd_proportional_perception": 1.0
            if self.rlsd_perception_credit_alloc
            in {"proportional", "prop", "likelihood_ratio", "lratio"}
            else 0.0,
            "credit/rlsd_softmax_beta": float(self.rlsd_softmax_beta),
            "credit/rlsd_weak_is_mask_image": 1.0 if self.rlsd_weak_mode in {"mask_image", "mask", "masked"} else 0.0,
            "credit/rlsd_random_patch_style": float(
                1.0 if self.rlsd_image_mask_style in {"random_patch", "patch_blacken", "rpb"} else 0.0
            ),
            "credit/rlsd_patch_h": float(self.rlsd_image_mask_patch_h),
            "credit/rlsd_patch_w": float(self.rlsd_image_mask_patch_w),
            "credit/rlsd_image_mask_ratio": float(self.rlsd_image_mask_ratio),
            "credit/rlsd_log_ratio_mean": float(np.mean(log_ratio_mean_vals)) if log_ratio_mean_vals else 0.0,
            "credit/rlsd_w_clip_low": float(self.rlsd_w_clip_low),
            "credit/rlsd_w_clip_high": float(self.rlsd_w_clip_high),
            "credit/rlsd_lambda_warmup_steps": float(self.rlsd_lambda_warmup_steps),
            "credit/rlsd_gate_k": float(self.rlsd_gate_k),
            "credit/simple_rlsd_gate_lr_m_bias": float(self.simple_rlsd_gate_lr_m_bias),
            "credit/simple_rlsd_teacher_b_gate_fixed": float(self.simple_rlsd_teacher_b_gate_fixed)
            if self.simple_rlsd_teacher_b_gate_fixed is not None
            else 0.0,
            "credit/simple_rlsd_teacher_b_gate_fixed_on": 1.0 if self.simple_rlsd_teacher_b_gate_fixed is not None else 0.0,
            "credit/rlsd_gate_pre_linear": float(np.mean(gate_pre_linear_vals)) if gate_pre_linear_vals else 0.0,
            "credit/simple_gate": float(np.mean(gate_vals)) if gate_vals else 0.0,
            "credit/simple_teacher_b_delta": float(np.mean(delta_vals)) if delta_vals else 0.0,
            "credit/simple_raw_weight_mean": float((raw_w * response_mask).sum().item() / max(1.0, response_mask.sum().item())),
            "credit/simple_raw_weight_mean_perception": raw_w_mean_perc,
            "credit/simple_raw_weight_mean_reasoning": raw_w_mean_reas,
            "credit/simple_mod_mean": float((modulation * response_mask).sum().item() / max(1.0, response_mask.sum().item())),
            "credit/simple_mod_mean_perception": mod_mean_perc,
            "credit/simple_mod_mean_reasoning": mod_mean_reas,
            "credit/simple_raw_w_max_perception": raw_w_max_perc,
            "credit/simple_raw_w_max_reasoning": raw_w_max_reas,
            "credit/simple_mod_max_perception": mod_max_perc,
            "credit/simple_mod_max_reasoning": mod_max_reas,
            "credit/simple_mod_lambda": float(self.simple_mod_lambda),
            "credit/simple_mod_clip": float(self.simple_mod_clip),
            "credit/rlsd_reward_delta_lambda": float(self.rlsd_reward_delta_lambda),
            "reward/rlsd_delta_shaping_mean": float((self.rlsd_reward_delta_lambda * rlsd_delta_bonus).mean().item()),
            "credit/teacher_b_gt_privilege": 1.0 if self.teacher_b_gt_privilege else 0.0,
            "credit/teacher_b_rollout_vs_qi": 1.0 if self.teacher_b_reasoning_delta_mode == "rollout_vs_qi" else 0.0,
        }

    def fit(self):
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)

                self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                self.actor_rollout_ref_wg.release_rollout_engine()

                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics_raw = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics_raw).items()})
                    else:
                        reward_metrics_raw = {"overall": batch.batch["token_level_scores"].sum(dim=-1).tolist()}

                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    metrics.update(self._compute_weighted_advantage(batch, reward_metrics_raw))

                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.non_tensor_batch))

                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.non_tensor_batch))

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.global_step > 0 and (
            self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0
        ):
            self._save_checkpoint()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")
