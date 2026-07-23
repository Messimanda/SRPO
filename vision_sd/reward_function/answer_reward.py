"""
Vision-SD answer reward.

Uses the final boxed answer for RL direction while the trainer supplies
token-level credit assignment from the dual-teacher mechanism.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "vision_sd_answer_reward"
REWARD_TYPE = "sequential"


def format_reward(response: str) -> float:
    pattern = re.compile(
        r"^\s*<perception>.*?</perception>\s*"
        r"<reasoning>.*?</reasoning>\s*"
        r"\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(response)
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.05) -> dict[str, float]:
    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]

    fmt = format_reward(response)
    acc = accuracy_reward(response, ground_truth)
    overall = (1.0 - format_weight) * acc + format_weight * fmt
    return {
        "overall": overall,
        "accuracy": acc,
        "format": fmt,
    }

