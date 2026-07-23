import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WINDOW = 20
ROOT = Path(__file__).resolve().parent
MODEL_DIRS = {
    "3B": ROOT / "3B",
    "7B": ROOT / "7B",
}
ALGO_ORDER = ["grpo", "dapo", "proposed"]
ALGO_LABEL = {
    "grpo": "GRPO",
    "dapo": "DAPO",
    "proposed": "SRPO",
}
ALGO_COLOR = {
    "grpo": "#2F5D8A",
    "dapo": "#C17C2F",
    "proposed": "#2E8B57",
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


def load_algo_curve(model_dir: Path, model_size: str, algo: str) -> pd.DataFrame:
    pattern = f"{model_size.lower()}-{algo}*.csv"
    files = sorted(model_dir.glob(pattern), key=get_chunk_index)
    if not files:
        raise FileNotFoundError(f"No files for {model_size} {algo} under {model_dir}")

    dfs = []
    offset = 0

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        metric_col = find_metric_column(df)
        cur = df[["Step", metric_col]].copy()
        cur = cur.rename(columns={"Step": "step", metric_col: "reward_acc"})
        cur = cur.dropna(subset=["step", "reward_acc"])

        if i == 0:
            # 第一段直接使用原始 Step
            cur["global_step"] = cur["step"].astype(float)
        else:
            # 后续段在上一段结尾后平移，并从 offset+1 开始，避免边界 step 重合
            base = float(cur["step"].iloc[0])
            cur["global_step"] = cur["step"].astype(float) - base + offset + 1

        offset = float(cur["global_step"].iloc[-1])
        dfs.append(cur[["global_step", "reward_acc"]])

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("global_step")
    out["running_avg"] = out["reward_acc"].rolling(window=WINDOW, min_periods=1).mean()
    return out


def plot_panel(ax, model_size: str):
    model_dir = MODEL_DIRS[model_size]

    for algo in ALGO_ORDER:
        curve = load_algo_curve(model_dir, model_size, algo)
        # Raw observations: translucent line + markers
        ax.plot(
            curve["global_step"],
            curve["reward_acc"],
            color=ALGO_COLOR[algo],
            linewidth=0.9,
            alpha=0.22,
            marker="o",
            markersize=2.6,
            markerfacecolor=ALGO_COLOR[algo],
            markeredgewidth=0.0,
            markevery=max(len(curve) // 45, 1),
        )
        # Smoothed trend
        ax.plot(
            curve["global_step"],
            curve["running_avg"],
            label=f"{ALGO_LABEL[algo]} {model_size}",
            color=ALGO_COLOR[algo],
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
        prop={"size": 11, "weight": "bold"},
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
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), sharey=False)

    plot_panel(axes[0], "3B")
    plot_panel(axes[1], "7B")

    plt.tight_layout()

    out_png = ROOT / "figure4_reward_accuracy.png"
    out_pdf = ROOT / "figure4_reward_accuracy.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
