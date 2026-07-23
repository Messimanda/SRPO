import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WINDOW = 20
ROOT = Path(__file__).resolve().parent

PROPOSED_DIR = {
    "3B": ROOT.parent / "reward_acc" / "3B",
    "7B": ROOT.parent / "reward_acc" / "7B",
}

SERIES_CONFIG = {
    "3B": {
        "proposed": (PROPOSED_DIR["3B"], "3b-proposed-*.csv", "SRPO"),
        "no_perception": (ROOT / "no_perception" / "3B", "3b-noperception-*.csv", "w/o Perception Credit"),
        "no_reasoning": (ROOT / "no_reasoning" / "3B", "3B-noreason-*.csv", "w/o Reasoning Credit"),
        "no_simplemod": (ROOT / "no_simplemod" / "3B", "3B-nosimplemod-*.csv", "w/o Unified Modulation"),
        "no_entropy": (ROOT / "entropy" / "3B" / "no_entrop", "3b-no-enrtopy-*.csv", r"$\eta$=0"),
        "entropy_01": (ROOT / "entropy" / "3B" / "0.1", "entropy-0.1-*.csv", r"$\eta$=0.1"),
    },
    "7B": {
        "proposed": (PROPOSED_DIR["7B"], "7b-proposed-*.csv", "SRPO"),
        "no_perception": (ROOT / "no_perception" / "7B", "7B-noperception-*.csv", "w/o Perception Credit"),
        "no_reasoning": (ROOT / "no_reasoning" / "7B", "7B-noreason-*.csv", "w/o Reasoning Credit"),
        "no_simplemod": (ROOT / "no_simplemod" / "7B", "7B-nosimple-*.csv", "w/o Unified Modulation"),
        "no_entropy": (ROOT / "entropy" / "7B" / "no_entropy", "7b-no-entropy-*.csv", r"$\eta$=0"),
        "entropy_01": (ROOT / "entropy" / "7B" / "0.1", "7B-entropy-0.1-*.csv", r"$\eta$=0.1"),
    },
}

COLOR = {
    "proposed": "#2E8B57",
    "no_perception": "#2F5D8A",
    "no_reasoning": "#C17C2F",
    "no_simplemod": "#B04A4A",
    "no_entropy": "#2F5D8A",
    "entropy_01": "#B04A4A",
}


def get_chunk_index(path: Path) -> int:
    m = re.search(r"-(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else 1


def find_metric_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "reward/accuracy" in c and not c.endswith("__MIN") and not c.endswith("__MAX"):
            return c
    raise ValueError(f"No reward/accuracy metric column found in {list(df.columns)}")


def load_curve(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(pattern), key=get_chunk_index)
    if not files:
        raise FileNotFoundError(f"No files matched: {data_dir / pattern}")

    chunks = []
    offset = 0.0
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        metric_col = find_metric_column(df)
        cur = df[["Step", metric_col]].copy()
        cur = cur.rename(columns={"Step": "step", metric_col: "reward_acc"}).dropna()
        if i == 0:
            cur["global_step"] = cur["step"].astype(float)
        else:
            base = float(cur["step"].iloc[0])
            cur["global_step"] = cur["step"].astype(float) - base + offset + 1
        offset = float(cur["global_step"].iloc[-1])
        chunks.append(cur[["global_step", "reward_acc"]])

    out = pd.concat(chunks, ignore_index=True).sort_values("global_step")
    out["running_avg"] = out["reward_acc"].rolling(window=WINDOW, min_periods=1).mean()
    return out


def plot_series(ax, model_size: str, keys: list[str], show_ylabel: bool):
    cfg = SERIES_CONFIG[model_size]
    entropy_panel = "no_entropy" in keys
    for key in keys:
        data_dir, pattern, label = cfg[key]
        if key == "proposed" and entropy_panel:
            label = r"$\eta$=0.03"
        curve = load_curve(data_dir, pattern)
        ax.plot(
            curve["global_step"],
            curve["reward_acc"],
            color=COLOR[key],
            linewidth=0.9,
            alpha=0.22,
            marker="o",
            markersize=2.6,
            markerfacecolor=COLOR[key],
            markeredgewidth=0.0,
            markevery=max(len(curve) // 45, 1),
        )
        ax.plot(
            curve["global_step"],
            curve["running_avg"],
            color=COLOR[key],
            linewidth=2.4,
            label=label,
        )

    ax.set_xlabel("Training Step", fontsize=16, fontweight="bold")
    ax.set_ylabel("Training accuracy" if show_ylabel else "", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_weight = "normal" if entropy_panel else "bold"
    leg = ax.legend(
        loc="lower right",
        frameon=True,
        prop={"size": 11, "weight": legend_weight},
        handlelength=2.2,
        fancybox=True,
        framealpha=0.25,
    )
    leg.get_frame().set_edgecolor((0, 0, 0, 0.35))


def plot_model(model_size: str):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), sharey=False)

    plot_series(
        axes[0],
        model_size,
        ["proposed", "no_perception", "no_reasoning", "no_simplemod"],
        show_ylabel=True,
    )
    plot_series(
        axes[1],
        model_size,
        ["proposed", "no_entropy", "entropy_01"],
        show_ylabel=False,
    )

    plt.tight_layout()
    out_png = ROOT / f"figure_ablation_dual_{model_size}.png"
    out_pdf = ROOT / f"figure_ablation_dual_{model_size}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 16,
            "figure.dpi": 300,
        }
    )
    plot_model("3B")
    plot_model("7B")


if __name__ == "__main__":
    main()
