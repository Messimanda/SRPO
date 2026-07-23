#!/usr/bin/env python3
"""
LLM Judge for evaluation responses.

1. Extracts the final answer from \\boxed{} in each model response (regex).
2. Sends the extracted answer + ground truth to an LLM judge (Qwen2.5-14B-Instruct
   via vLLM) to determine correctness.

This keeps prompts very short (just two short strings to compare), avoiding
context-window issues with long chain-of-thought reasoning.

Usage:
    python evaluation/llm_judge.py --response_dir ./evaluation/responses/vision_r1_full/Qwen2.5-VL-7B-Instruct

    python evaluation/llm_judge.py --response_file ./evaluation/responses/vision_r1_full/Qwen2.5-VL-7B-Instruct/mmstar.jsonl

    python evaluation/llm_judge.py --response_dir ... --judge_model Qwen/Qwen2.5-14B-Instruct --tp_size 4
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import time
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from mathruler.grader import extract_boxed_content


JUDGE_PROMPT_TEMPLATE = """\
You are an expert answer evaluator. Determine whether the model's extracted answer is correct by comparing it to the ground truth.

## Model's Extracted Answer:
{extracted_answer}

## Ground Truth Answer:
{ground_truth}

## Instructions:
Compare the two answers. Be flexible with formatting differences (e.g. "1/2" vs "0.5", "A" vs "(A)", equivalent expressions) but strict on mathematical/factual correctness.
Return your judgment as a JSON object with EXACTLY this format (no other text):

{{"verdict": "correct" or "incorrect", "reasoning": "<brief explanation>"}}
"""


def extract_boxed(response: str) -> str:
    """Extract content from \\boxed{} in the response."""
    try:
        answer = extract_boxed_content(response)
        return answer if answer else ""
    except Exception:
        return ""


def build_judge_messages(extracted_answer: str, ground_truth: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a precise answer evaluation assistant. Always respond with valid JSON only."},
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
            extracted_answer=extracted_answer if extracted_answer else "(no answer extracted)",
            ground_truth=ground_truth,
        )},
    ]


def parse_judge_output(text: str) -> dict[str, Any]:
    """Extract the JSON verdict from the judge's response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {"verdict": "incorrect", "reasoning": f"Failed to parse judge output: {text[:200]}"}


def collect_response_files(response_dir: str | None, response_file: str | None) -> list[str]:
    files = []
    if response_file:
        files.append(response_file)
    if response_dir:
        pattern = os.path.join(response_dir, "**", "*.jsonl")
        files.extend(sorted(glob.glob(pattern, recursive=True)))
        if not files:
            pattern = os.path.join(response_dir, "*.jsonl")
            files.extend(sorted(glob.glob(pattern)))
    return files


def response_path_to_judgment_path(response_path: str) -> str:
    """Mirror the response path structure under evaluation/judgments/."""
    p = Path(response_path)
    parts = list(p.parts)
    try:
        idx = parts.index("responses")
        parts[idx] = "judgments"
    except ValueError:
        parts.insert(-1, "judgments")
    return str(Path(*parts))


def load_responses(filepath: str) -> list[dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def call_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")

    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {408, 429, 500, 502, 503, 504}
            last_err = RuntimeError(f"OpenAI API error {exc.code}: {err_body[:500]}")
            if not should_retry or attempt == max_retries - 1:
                raise last_err
        except Exception as exc:
            last_err = exc
            if attempt == max_retries - 1:
                raise

        sleep_s = min(2 ** attempt, 30)
        time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI API call failed after retries: {last_err}")


def judge_with_openai(
    records: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
    concurrency: int,
) -> list[str]:
    outputs = [""] * len(records)

    def run_one(item: tuple[int, dict[str, Any]]) -> tuple[int, str]:
        idx, rec = item
        messages = build_judge_messages(rec["_extracted_answer"], rec["ground_truth"])
        text = call_openai_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        return idx, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(run_one, item) for item in enumerate(records)]
        for future in concurrent.futures.as_completed(futures):
            idx, text = future.result()
            outputs[idx] = text

    return outputs


def main():
    parser = argparse.ArgumentParser(description="LLM Judge: extract \\boxed{} then judge with LLM")
    parser.add_argument("--response_dir", type=str, default=None,
                        help="Directory containing response JSONL files")
    parser.add_argument("--response_file", type=str, default=None,
                        help="Single response JSONL file to judge")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Judge model name or path")
    parser.add_argument("--judge_backend", type=str, default="auto",
                        choices=["auto", "vllm", "openai"],
                        help="Judge backend. 'auto' uses OpenAI for gpt-* / o* models, otherwise vLLM.")
    parser.add_argument("--tp_size", type=int, default=None,
                        help="Tensor parallel size (default: auto-detect GPU count)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens for judge response")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for judge")
    parser.add_argument("--openai_timeout", type=float, default=120.0,
                        help="Timeout in seconds for each OpenAI API request")
    parser.add_argument("--openai_max_retries", type=int, default=6,
                        help="Maximum retries for each OpenAI API request")
    parser.add_argument("--openai_concurrency", type=int, default=8,
                        help="Concurrent OpenAI API requests")
    args = parser.parse_args()

    if not args.response_dir and not args.response_file:
        parser.error("Provide --response_dir or --response_file")

    response_files = collect_response_files(args.response_dir, args.response_file)
    if not response_files:
        print("No response JSONL files found.")
        sys.exit(1)

    print(f"Found {len(response_files)} response file(s) to judge.")

    # --- Step 1: Extract \\boxed{} from all responses ---
    all_file_records: list[tuple[str, list[dict]]] = []
    total_samples = 0
    for resp_file in response_files:
        records = load_responses(resp_file)
        for rec in records:
            rec["_extracted_answer"] = extract_boxed(rec["response"])
        all_file_records.append((resp_file, records))
        total_samples += len(records)

    print(f"Extracted \\boxed{{}} answers from {total_samples} total samples.")

    judge_backend = args.judge_backend
    if judge_backend == "auto":
        model_name = args.judge_model.lower()
        if model_name.startswith("gpt-") or model_name.startswith("o"):
            judge_backend = "openai"
        else:
            judge_backend = "vllm"

    llm = None
    tokenizer = None
    sampling_params = None
    if judge_backend == "vllm":
        tp_size = args.tp_size or torch.cuda.device_count()
        if tp_size < 1:
            tp_size = 1
        print(f"Loading vLLM judge model: {args.judge_model} (tp={tp_size})")

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.judge_model,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_model_len=4096,
        )
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        print(f"Using OpenAI judge model: {args.judge_model}")

    # --- Step 3: Judge each file ---
    for resp_file, records in all_file_records:
        print(f"\n{'='*60}")
        print(f"Judging: {resp_file}")
        print(f"{'='*60}")

        if not records:
            print("  (empty file, skipping)")
            continue

        print(f"  Running {judge_backend} judge on {len(records)} samples...")
        if judge_backend == "vllm":
            prompts = []
            for rec in records:
                messages = build_judge_messages(rec["_extracted_answer"], rec["ground_truth"])
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_text)
            model_outputs = llm.generate(prompts, sampling_params)
            judge_texts = [output.outputs[0].text for output in model_outputs]
        else:
            judge_texts = judge_with_openai(
                records=records,
                model=args.judge_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.openai_timeout,
                max_retries=args.openai_max_retries,
                concurrency=args.openai_concurrency,
            )

        judgments = []
        correct_count = 0
        for rec, judge_text in zip(records, judge_texts):
            parsed = parse_judge_output(judge_text)

            judgment = {
                "prompt": rec.get("prompt", ""),
                "response": rec["response"],
                "ground_truth": rec["ground_truth"],
                "extracted_answer": rec["_extracted_answer"],
                "judge_verdict": parsed.get("verdict", "incorrect"),
                "judge_reasoning": parsed.get("reasoning", ""),
                "original_score": rec.get("score", None),
                "judge_backend": judge_backend,
                "judge_model": args.judge_model,
            }
            judgments.append(judgment)
            if judgment["judge_verdict"] == "correct":
                correct_count += 1

        judgment_path = response_path_to_judgment_path(resp_file)
        os.makedirs(os.path.dirname(judgment_path), exist_ok=True)
        with open(judgment_path, "w", encoding="utf-8") as f:
            for j in judgments:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

        acc = correct_count / len(judgments) * 100 if judgments else 0.0
        dataset_name = Path(resp_file).stem
        print(f"  {dataset_name}: {correct_count}/{len(judgments)} correct ({acc:.1f}%)")
        print(f"  Saved to: {judgment_path}")

    print("\nDone. All judgments saved.")


if __name__ == "__main__":
    main()
