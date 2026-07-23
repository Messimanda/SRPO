#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WINDOW = 20
ROOT = Path(__file__).resolve().parent
NO_SIMPLEMOD_DIR = ROOT / "ablation" / "no_simplemod" / "3B"
PROPOSED_DIR = ROOT / "reward_acc" / "3B"

CURVES = {
    "proposed": {
        "dir": PROPOSED_DIR,
        "pattern": "3b-proposed-*.csv",
        "label": "Proposed",
        "color": "#2E8B57",
    },
    "nosimplemod": {
        "dir": NO_SIMPLEMOD_DIR,
        "pattern": "3B-nosimplemod-*.csv",
        "label": "No-SimpleMod",
        "color": "#B04A4A",
    },
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

        if i == 0:
            cur["global_step"] = cur["step"].astype(float)
        else:
            base = float(cur["step"].iloc[0])
            cur["global_step"] = cur["step"].astype(float) - base + offset + 1

        offset = float(cur["global_step"].iloc[-1])
        dfs.append(cur[["global_step", "reward_acc"]])

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("global_step")
    out["running_avg"] = out["reward_acc"].rolling(window=WINDOW, min_periods=1).mean()
    return out


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 16,
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.1))

    for key in ["proposed", "nosimplemod"]:
        spec = CURVES[key]
        curve = load_curve(spec["dir"], spec["pattern"])

        ax.plot(
            curve["global_step"],
            curve["reward_acc"],
            color=spec["color"],
            linewidth=0.9,
            alpha=0.22,
            marker="o",
            markersize=2.6,
            markerfacecolor=spec["color"],
            markeredgewidth=0.0,
            markevery=max(len(curve) // 45, 1),
        )

        ax.plot(
            curve["global_step"],
            curve["running_avg"],
            label=f"{spec['label']} 3B",
            color=spec["color"],
            linewidth=2.4,
        )

    ax.set_xlabel("Training Step", fontsize=16)
    ax.set_ylabel("Training accuracy", fontsize=16)
    ax.grid(alpha=0.22, linewidth=0.8)
    leg = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=11,
        handlelength=2.2,
        fancybox=True,
        framealpha=0.25,
    )
    leg.get_frame().set_edgecolor((0, 0, 0, 0.35))
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_png = ROOT / "ablation" / "no_simplemod" / "3B" / "figure_nosimplemod_vs_proposed_3B.png"
    out_pdf = ROOT / "ablation" / "no_simplemod" / "3B" / "figure_nosimplemod_vs_proposed_3B.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
