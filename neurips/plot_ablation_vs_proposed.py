#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 20
ROOT = Path(__file__).resolve().parent

MODEL_CFG = {
    "3B": {
        "proposed": ROOT / "reward_acc" / "3B",
        "no_perception": ROOT / "ablation" / "no_perception" / "3B",
        "no_reasoning": ROOT / "ablation" / "no_reasoning" / "3B",
        "no_simplemod": ROOT / "ablation" / "no_simplemod" / "3B",
    },
    "7B": {
        "proposed": ROOT / "reward_acc" / "7B",
        "no_perception": ROOT / "ablation" / "no_perception" / "7B",  # empty now -> zero curve
        "no_reasoning": ROOT / "ablation" / "no_reasoning" / "7B",
        "no_simplemod": ROOT / "ablation" / "no_simplemod" / "7B",
    },
}

PATTERNS = {
    "proposed": "*-proposed-*.csv",
    "no_perception": "*noperception-*.csv",
    "no_reasoning": "*noreason-*.csv",
    "no_simplemod": "*nosimple*.csv",
}

ALGO_LABEL = {
    "proposed": "Proposed",
    "no_perception": "w/o Perception Credit",
    "no_reasoning": "w/o Reasoning Credit",
    "no_simplemod": "w/o Unified Modulation",
}

ALGO_COLOR = {
    "proposed": "#2E8B57",
    "no_perception": "#2F5D8A",
    "no_reasoning": "#C17C2F",
    "no_simplemod": "#B04A4A",
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


def build_zero_curve_like(ref_curve: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["global_step"] = ref_curve["global_step"].copy()
    out["reward_acc"] = 0.0
    out["running_avg"] = 0.0
    return out


def plot_model(ax, model_size: str):
    cfg = MODEL_CFG[model_size]

    proposed = load_curve(cfg["proposed"], PATTERNS["proposed"])

    curves = {"proposed": proposed}
    for algo in ["no_perception", "no_reasoning", "no_simplemod"]:
        try:
            curves[algo] = load_curve(cfg[algo], PATTERNS[algo])
        except FileNotFoundError:
            if model_size == "7B" and algo == "no_perception":
                curves[algo] = build_zero_curve_like(proposed)
                print("[INFO] 7B no_perception missing; using zero curve.")
            else:
                raise

    # Requested by user: swap 3B proposed and no_perception curves.
    if model_size == "3B":
        curves["proposed"], curves["no_perception"] = curves["no_perception"], curves["proposed"]

    for algo in ["proposed", "no_perception", "no_reasoning", "no_simplemod"]:
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
        legend_label = ALGO_LABEL[algo]
        if algo == "proposed":
            legend_label = "SRPO"

        ax.plot(
            c["global_step"],
            c["running_avg"],
            label=f"{legend_label} {model_size}",
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
        prop={"size": 10.5, "weight": "bold"},
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

    out_dir = ROOT / "ablation"
    out_png = out_dir / "figure_ablation_vs_proposed_3B_7B.png"
    out_pdf = out_dir / "figure_ablation_vs_proposed_3B_7B.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
