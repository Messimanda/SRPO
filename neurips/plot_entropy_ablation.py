#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 20
ROOT = Path(__file__).resolve().parent

MODEL_CFG = {
    "3B": {
        "no_entropy": ROOT / "ablation" / "entropy" / "3B" / "no_entrop",
        "entropy_01": ROOT / "ablation" / "entropy" / "3B" / "0.1",
        "entropy_003": ROOT / "reward_acc" / "3B",
    },
    "7B": {
        "no_entropy": ROOT / "ablation" / "entropy" / "7B" / "no_entropy",
        "entropy_01": ROOT / "ablation" / "entropy" / "7B" / "0.1",
        "entropy_003": ROOT / "reward_acc" / "7B",
    },
}

PATTERNS = {
    "no_entropy": "*.csv",
    "entropy_01": "*.csv",
    "entropy_003": "*-proposed-*.csv",
}

ALGO_LABEL = {
    "no_entropy": r"$\eta$=0",
    "entropy_003": r"$\eta$=0.03",
    "entropy_01": r"$\eta$=0.1",
}

ALGO_COLOR = {
    "no_entropy": "#2F5D8A",
    "entropy_003": "#2E8B57",
    "entropy_01": "#B04A4A",
}


def get_chunk_index(path: Path) -> int:
    m = re.search(r"-(\d+)\.csv$", path.name)
    if m:
        return int(m.group(1))
    return 1


def find_metric_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "reward/accuracy" in c and not c.endswith("__MIN") and not c.endswith("__MAX"):
            return c
    raise ValueError(f"No reward/accuracy metric column found in {list(df.columns)}")


def load_curve(csv_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(csv_dir.glob(pattern), key=get_chunk_index)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {csv_dir}")

    dfs = []
    offset = 0.0

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        metric_col = find_metric_column(df)
        cur = df[["Step", metric_col]].copy()
        cur = cur.rename(columns={"Step": "step", metric_col: "reward_acc"})
        cur = cur.dropna(subset=["step", "reward_acc"])

        if cur.empty:
            continue

        if i == 0:
            cur["global_step"] = cur["step"].astype(float)
        else:
            base = float(cur["step"].iloc[0])
            cur["global_step"] = cur["step"].astype(float) - base + offset + 1

        offset = float(cur["global_step"].iloc[-1])
        dfs.append(cur[["global_step", "reward_acc"]])

    if not dfs:
        raise ValueError(f"All files empty for {pattern} under {csv_dir}")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("global_step")
    out["running_avg"] = out["reward_acc"].rolling(window=WINDOW, min_periods=1).mean()
    return out


def plot_model(ax, model_size: str):
    cfg = MODEL_CFG[model_size]

    curves = {}
    for algo in ["no_entropy", "entropy_003", "entropy_01"]:
        curves[algo] = load_curve(cfg[algo], PATTERNS[algo])

    for algo in ["no_entropy", "entropy_003", "entropy_01"]:
        c = curves[algo]
        color = ALGO_COLOR[algo]

        ax.plot(
            c["global_step"],
            c["reward_acc"],
            color=color,
            linewidth=0.9,
            alpha=0.22,
            marker="o",
            markersize=2.6,
            markerfacecolor=color,
            markeredgewidth=0.0,
            markevery=max(len(c) // 45, 1),
        )

        ax.plot(
            c["global_step"],
            c["running_avg"],
            label=f"{ALGO_LABEL[algo]} " + rf"$\mathbf{{{model_size}}}$",
            color=color,
            linewidth=2.4,
        )

    ax.set_xlabel("Training Step", fontsize=16, fontweight="bold")
    if model_size == "3B":
        ax.set_ylabel("Training accuracy", fontsize=16, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.grid(alpha=0.22, linewidth=0.8)
    leg = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=10.5,
        handlelength=2.2,
        fancybox=True,
        framealpha=0.25,
    )
    leg.get_frame().set_edgecolor((0, 0, 0, 0.35))
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 16,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=False)

    plot_model(axes[0], "3B")
    plot_model(axes[1], "7B")

    plt.tight_layout()

    out_dir = ROOT / "ablation" / "entropy"
    out_png = out_dir / "figure_entropy_ablation_3B_7B.png"
    out_pdf = out_dir / "figure_entropy_ablation_3B_7B.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
