#!/usr/bin/env python3
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 20


def segment_index(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"-(\d+)\.csv$", name)
    return int(m.group(1)) if m else 10**9


def pick_accuracy_column(df: pd.DataFrame) -> str:
    cols = [
        c for c in df.columns
        if c != "Step" and not c.endswith("__MIN") and not c.endswith("__MAX")
    ]
    if not cols:
        raise ValueError("No accuracy column found (excluding Step/__MIN/__MAX).")
    return cols[0]


def merge_interrupted_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern), key=segment_index)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    merged_parts = []
    current_max_step = 0

    for f in files:
        df = pd.read_csv(f)
        if "Step" not in df.columns:
            raise ValueError(f"Missing 'Step' column in {f}")

        acc_col = pick_accuracy_column(df)
        part = df[["Step", acc_col]].copy()
        part.columns = ["Step", "Accuracy"]
        part["Step"] = pd.to_numeric(part["Step"], errors="coerce")
        part["Accuracy"] = pd.to_numeric(part["Accuracy"], errors="coerce")
        part = part.dropna().sort_values("Step").reset_index(drop=True)

        if part.empty:
            continue

        min_step = int(part["Step"].min())
        if min_step <= current_max_step:
            offset = current_max_step - min_step + 1
            part["Step"] = part["Step"] + offset

        current_max_step = int(part["Step"].max())
        merged_parts.append(part)

    if not merged_parts:
        raise ValueError(f"All matched files are empty or invalid for pattern: {pattern}")

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values("Step").drop_duplicates(subset=["Step"], keep="last")
    merged = merged.reset_index(drop=True)
    merged["RunningAvg"] = merged["Accuracy"].rolling(window=WINDOW, min_periods=1).mean()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot no_simplemod vs proposed training accuracy for 7B model."
    )
    parser.add_argument(
        "--nosimple-pattern",
        default="/data/250010207/Vision-SR1/neurips/ablation/no_simplemod/7B/7B-nosimple-*.csv",
        help="Glob pattern for interrupted no_simplemod CSV segments.",
    )
    parser.add_argument(
        "--proposed-pattern",
        default="/data/250010207/Vision-SR1/neurips/reward_acc/7B/7b-proposed-*.csv",
        help="Glob pattern for interrupted proposed CSV segments.",
    )
    parser.add_argument(
        "--output",
        default="/data/250010207/Vision-SR1/neurips/ablation/no_simplemod/7B/7b_nosimplemod_vs_proposed.png",
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI.")
    args = parser.parse_args()

    no_simple = merge_interrupted_csvs(args.nosimple_pattern)
    proposed = merge_interrupted_csvs(args.proposed_pattern)

    plt.figure(figsize=(10, 6))
    plt.plot(
        no_simple["Step"],
        no_simple["Accuracy"],
        color="#C17C2F",
        linewidth=0.9,
        alpha=0.22,
        marker="o",
        markersize=2.6,
        markerfacecolor="#C17C2F",
        markeredgewidth=0.0,
        markevery=max(len(no_simple) // 45, 1),
    )
    plt.plot(
        proposed["Step"],
        proposed["Accuracy"],
        color="#2E8B57",
        linewidth=0.9,
        alpha=0.22,
        marker="o",
        markersize=2.6,
        markerfacecolor="#2E8B57",
        markeredgewidth=0.0,
        markevery=max(len(proposed) // 45, 1),
    )
    plt.plot(
        no_simple["Step"],
        no_simple["RunningAvg"],
        label=f"No SimpleMod (MA{WINDOW})",
        color="#C17C2F",
        linewidth=2.4,
    )
    plt.plot(
        proposed["Step"],
        proposed["RunningAvg"],
        label=f"Proposed (MA{WINDOW})",
        color="#2E8B57",
        linewidth=2.4,
    )

    plt.title("7B Training Accuracy: No SimpleMod vs Proposed")
    plt.xlabel("Training Step")
    plt.ylabel("Reward Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
