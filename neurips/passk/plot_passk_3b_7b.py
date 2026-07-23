from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
K_VALUES = [1, 2, 4, 8]

DATA = {
    "3B": {
        "mathvision": {
            "DAPO-3B": [25.65, 35.46052631578947, 50.31578947368421, 59.1578947368421],
            "PRCO-3B": [25.32, 37.828947368421055, 48.35526315789474, 56.9078947368421],
            "PAPO-3B": [25.98, 35.85526315789474, 45.39473684210526, 55.26315789473684],
            "SRPO-3B": [29.65, 40.85526315789474, 51.328947368421055, 63.53947368421053],
        },
        "mmupro": {
            "DAPO-3B": [28.15, 34.797687861271676, 42.202312138728324, 54.31791907514451],
            "PRCO-3B": [27.8, 34.104046242774565, 40.86705202312139, 49.36416184971098],
            "PAPO-3B": [27.91, 33.641618497109825, 41.96531791907515, 51.73410404624278],
            "SRPO-3B": [29.27, 35.83236994219653, 45.89017341040462, 55.21965317919075],
        },
    },
    "7B": {
        "mathvision": {
            "DAPO-7B": [29.27, 37.5, 49.671052631578945, 58.55263157894737],
            "VPPO-7B": [30.26, 40.78947368421053, 50.0, 59.53947368421053],
            "PRCO-7B": [30.92, 37.828947368421055, 50.2, 61.0421052631579],
            "PAPO-7B": [29.93, 40.13157894736842, 50.31578947368421, 59.86842105263158],
            "SRPO-7B": [34.53947368421053, 41.44736842105263, 51.6578947368421, 62.171052631578945],
        },
        "mmupro": {
            "DAPO-7B": [35.2, 39.71098265895954, 44.21965317919075, 47.6878612716763],
            "VPPO-7B": [35.89, 41.12138728323699, 48.17341040462428, 51.589595375722546],
            "PRCO-7B": [36.93, 40.809248554913296, 46.127167630057805, 51.84971098265896],
            "PAPO-7B": [35.2, 39.76878612716763, 45.14450867052023, 47.884393063583815],
            "SRPO-7B": [38.09, 43.93063583815029, 49.884393063583815, 53.01156069364162],
        },
    },
}

ALGO_COLOR = {
    "DAPO": "#C17C2F",
    "VPPO": "#2F5D8A",
    "PRCO": "#7B4EA3",
    "PAPO": "#8C564B",
    "SRPO": "#2E8B57",
}
DATASET_STYLE = {
    "mathvision": "-",
    "mmupro": "--",
}
DATASET_LABEL = {
    "mathvision": "MathVision",
    "mmupro": "MMMU-Pro",
}


def algo_key(name: str) -> str:
    return name.split("-")[0]


def plot_panel(ax, model_size: str, dataset: str, show_ylabel: bool, panel_tag: str):
    algo_scores = DATA[model_size][dataset]
    legend_order = []

    for algo_name, scores in algo_scores.items():
        key = algo_key(algo_name)
        legend_order.append(f"{key}-{model_size}")
        ax.plot(
            K_VALUES,
            scores,
            DATASET_STYLE[dataset],
            color=ALGO_COLOR.get(key, "#444444"),
            linewidth=2.2,
            marker="o",
            markersize=4.2,
            label=f"{key}-{model_size}",
        )

    ax.set_xlabel("Number of Responses Sampled (k)", fontsize=13, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.5,
        -0.28,
        f"{panel_tag} {DATASET_LABEL[dataset]}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
    )

    handles, labels = ax.get_legend_handles_labels()
    uniq = {lb: h for h, lb in zip(handles, labels)}
    ordered_labels = [lb for lb in legend_order if lb in uniq]
    ordered_handles = [uniq[lb] for lb in ordered_labels]
    leg = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="lower right",
        frameon=True,
        prop={"size": 9.5, "weight": "bold"},
        handlelength=2.0,
        fancybox=True,
        framealpha=0.25,
        ncol=1,
    )
    leg.get_frame().set_edgecolor((0, 0, 0, 0.35))


def main():
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 14,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.4), sharey=True)
    plot_panel(axes[0], "3B", "mathvision", show_ylabel=True, panel_tag="(a)")
    plot_panel(axes[1], "3B", "mmupro", show_ylabel=False, panel_tag="(b)")
    plot_panel(axes[2], "7B", "mathvision", show_ylabel=False, panel_tag="(c)")
    plot_panel(axes[3], "7B", "mmupro", show_ylabel=False, panel_tag="(d)")

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    out_png = ROOT / "passk_3b_7b.png"
    out_pdf = ROOT / "passk_3b_7b.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
