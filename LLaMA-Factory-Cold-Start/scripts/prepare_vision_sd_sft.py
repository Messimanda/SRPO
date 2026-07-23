#!/usr/bin/env python3
"""Prepare Vision-SD-aligned cold-start SFT datasets.

This converts the original Vision-SR1 cold-start data into the structured
format expected by the new Vision-SD algorithm:

<perception> ... </perception> <reasoning> ... </reasoning> \boxed{...}
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SOURCE_VISION = DATA_DIR / "A-vision_see_think.json"
SOURCE_TEXT = DATA_DIR / "A-caption_reasoners(text).json"

TARGET_VISION = DATA_DIR / "A-vision_sd_see_reason_answer.json"
TARGET_TEXT = DATA_DIR / "A-caption_perception_reason_answer.json"


VISION_PROMPT = (
    "Question: {question}\n"
    "-\n"
    "Analyze the image/video and solve the problem with an explicit decomposition. "
    "First, write a concise but self-contained perception block that captures only the evidence needed for solving "
    "the question. Prefer evidence units such as objects, counts, relations, text/OCR, and attributes. "
    "Wrap this part in <perception> </perception> tags.\n"
    "Next, write the reasoning process grounded in the image/video and the perception block, and wrap it in "
    "<reasoning> </reasoning> tags.\n"
    "Finally, give the answer in \\boxed{{}}.\n"
    "The output format must be: <perception> visual evidence here </perception> "
    "<reasoning> grounded reasoning here </reasoning> \\boxed{{FINAL ANSWER}}.<image>"
)


TEXT_PROMPT = (
    "Text evidence: {description}\n"
    "Question: {question}\n"
    "-\n"
    "You are given a text description instead of the original image/video. "
    "First, rewrite the useful evidence as a concise <perception> block using evidence units only. "
    "Then produce a grounded <reasoning> block based on that perception and answer the question in \\boxed{{}}.\n"
    "The output format must be: <perception> evidence here </perception> "
    "<reasoning> grounded reasoning here </reasoning> \\boxed{{FINAL ANSWER}}."
)


DESCRIPTION_PATTERN = re.compile(r"<description>([\s\S]*?)</description>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>([\s\S]*?)</think>", re.DOTALL)
PERCEPTION_PATTERN = re.compile(r"<perception>([\s\S]*?)</perception>", re.DOTALL)
REASONING_PATTERN = re.compile(r"<reasoning>([\s\S]*?)</reasoning>", re.DOTALL)
BOXED_PATTERN = re.compile(r"(\\boxed\{[\s\S]*\})", re.DOTALL)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload):
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_question_from_vision_prompt(prompt: str) -> str:
    text = prompt.strip()
    if text.endswith("<image>"):
        text = text[: -len("<image>")].rstrip()

    if text.startswith("Question: "):
        text = text[len("Question: ") :]

    # Handle known legacy templates from Vision-SR1 cold-start prompts.
    cut_markers = [
        "\n-\nYou are tasked with analyzing an image/video",
        "\nYou are tasked with analyzing an image/video",
        "\n-\nYou FIRST think about the reasoning process",
        "\nYou FIRST think about the reasoning process",
        "\n-\nFirst analyze the image/video",
        "\nFirst analyze the image/video",
    ]
    for marker in cut_markers:
        if marker in text:
            return text.split(marker, 1)[0].strip()

    # Fallback: cut at the first instruction-like line after the question body.
    match = re.search(r"\n(?:You|First|Please|Output)\b", text)
    if match is not None:
        return text[: match.start()].strip()
    return text.strip()


def extract_text_description_and_question(prompt: str) -> tuple[str, str]:
    description_marker = "Text description: "
    question_marker = "\nQuestion: "
    task_marker = "\nYou are provided a text description"

    if not prompt.startswith(description_marker):
        raise ValueError("Unexpected caption_reasoners prompt format.")

    description_and_rest = prompt[len(description_marker) :]
    description, rest = description_and_rest.split(question_marker, 1)
    if task_marker in rest:
        question = rest.split(task_marker, 1)[0].strip()
    else:
        question = rest.strip()
    return description.strip(), question


def convert_vision_output(content: str) -> str:
    description_match = DESCRIPTION_PATTERN.search(content)
    reasoning_match = THINK_PATTERN.search(content)
    answer_match = BOXED_PATTERN.search(content)
    if description_match is None or reasoning_match is None or answer_match is None:
        if description_match is None or answer_match is None:
            raise ValueError("Unexpected vision output format.")
        answer = answer_match.group(1).strip()
        reasoning = content.split(answer, 1)[0]
        reasoning = reasoning.replace(description_match.group(0), "", 1).replace("<think>", "").replace("</think>", "").strip()
        perception = description_match.group(1).strip()
        return f"<perception>{perception}</perception>\n<reasoning>{reasoning}</reasoning>\n{answer}"

    perception = description_match.group(1).strip()
    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip()
    return f"<perception>{perception}</perception>\n<reasoning>{reasoning}</reasoning>\n{answer}"


def convert_text_output(content: str, description: str) -> str:
    reasoning_match = THINK_PATTERN.search(content)
    answer_match = BOXED_PATTERN.search(content)
    if answer_match is None:
        raise ValueError("Unexpected caption output format.")

    answer = answer_match.group(1).strip()
    if reasoning_match is not None:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = content.split(answer, 1)[0].replace("<think>", "").replace("</think>", "").strip()
    perception = description.strip()
    return f"<perception>{perception}</perception>\n<reasoning>{reasoning}</reasoning>\n{answer}"


def prepare_vision_dataset():
    source = load_json(SOURCE_VISION)
    converted = []
    for item in source:
        user = item["messages"][0]["content"]
        assistant = item["messages"][1]["content"]
        question = extract_question_from_vision_prompt(user)
        converted.append(
            {
                "messages": [
                    {"role": "user", "content": VISION_PROMPT.format(question=question)},
                    {"role": "assistant", "content": convert_vision_output(assistant)},
                ],
                "images": item.get("images", []),
            }
        )
    dump_json(TARGET_VISION, converted)
    return len(converted)


def prepare_text_dataset():
    source = load_json(SOURCE_TEXT)
    converted = []
    for item in source:
        description, question = extract_text_description_and_question(item["instruction"])
        converted.append(
            {
                "instruction": TEXT_PROMPT.format(description=description, question=question),
                "input": item.get("input", ""),
                "output": convert_text_output(item["output"], description),
            }
        )
    dump_json(TARGET_TEXT, converted)
    return len(converted)


def main():
    vision_count = prepare_vision_dataset()
    text_count = prepare_text_dataset()
    print(
        f"Prepared Vision-SD SFT datasets: {TARGET_VISION.name} ({vision_count} samples), "
        f"{TARGET_TEXT.name} ({text_count} samples)."
    )


if __name__ == "__main__":
    main()
