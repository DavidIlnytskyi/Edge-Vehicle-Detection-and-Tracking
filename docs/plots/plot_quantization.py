"""Plot the saved trained-model precision experiments without rerunning inference.

Run from any directory: python docs/plots/plot_quantization.py
Requires Matplotlib. Outputs quantization.png and quantization.svg beside this file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    source = output_dir.parent / "quantization.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["model"]: row
            for row in csv.DictReader(handle)
            if row["group"] == "my_trained"
        }

    formats = ["FP32", "FP32_pruned", "FP16", "INT8_static"]
    labels = ["FP32", "FP32\npruning experiment", "FP16", "INT8 static"]
    latency = [float(rows[name]["inference_ms"]) for name in formats]
    map50 = [100 * float(rows[name]["mAP50"]) for name in formats]
    fp16_index = formats.index("FP16")
    int8_index = formats.index("INT8_static")

    ink, muted = "#182D40", "#5B6D7C"
    green, blue, neutral = "#15775E", "#267AB3", "#8295A6"
    colors = [neutral, neutral, green, blue]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": ink,
        "axes.labelcolor": muted,
        "xtick.color": muted,
        "ytick.color": ink,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.19, right=0.96, top=0.70, bottom=0.27, wspace=0.30)
    fig.text(0.055, 0.935, "Recorded precision experiments", fontsize=21, weight="bold")
    for index, y in [(fp16_index, 0.865), (int8_index, 0.815)]:
        speedup = latency[0] / latency[index]
        accuracy_change = map50[index] - map50[0]
        fig.text(
            0.055, y,
            f"{labels[index]}: {speedup:.2f}× speedup · "
            f"{accuracy_change:+.3f} percentage points mAP@50",
            fontsize=13, weight="bold", color=colors[index],
        )

    for ax, values, title, unit, limit in zip(
        axes,
        [latency, map50],
        ["Reported inference latency", "Detection accuracy"],
        ["Milliseconds · lower is better", "mAP@50 (%) · higher is better"],
        [max(latency) * 1.30, max(map50) * 1.30],
    ):
        ax.axhspan(fp16_index - 0.42, fp16_index + 0.42, color="#EDF7F2", zorder=0)
        ax.axhspan(int8_index - 0.42, int8_index + 0.42, color="#EDF5FB", zorder=0)
        ax.barh(range(len(formats)), values, color=colors, height=0.49, zorder=3)
        ax.set_xlim(0, limit)
        ax.set_yticks(range(len(formats)), labels=labels)
        ax.set_title(title, loc="left", weight="bold", fontsize=12, pad=13)
        ax.set_xlabel(unit, labelpad=10, fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E6EBEF", linewidth=0.8)
        ax.tick_params(axis="both", length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for index, value in enumerate(values):
            text = f"{value:.2f}" if ax is axes[0] else f"{value:.3f}%"
            if value == 0:
                ax.plot(0, index, marker="|", markersize=17, color=colors[index], clip_on=False)
            ax.text(
                value + limit * 0.025, index, text,
                va="center", fontsize=10,
                color=colors[index] if index >= 2 else ink,
                weight="bold" if index in (fp16_index, int8_index) else "normal",
            )
    axes[0].invert_yaxis()

    fig.text(
        0.055, 0.12,
        "Saved experiment · my_trained group · NVIDIA T4 · evaluation split and runtime unspecified.",
        fontsize=10, color=muted,
    )
    fig.text(
        0.055, 0.076,
        "Source: docs/quantization.csv. CSV precision = detection precision; model = numeric format.",
        fontsize=9, color=muted,
    )
    fig.text(
        0.055, 0.035,
        "Speedups and accuracy changes are relative to FP32 in this saved experiment.",
        fontsize=9, color=muted,
    )
    fig.savefig(output_dir / "quantization.png", dpi=180, facecolor="white")
    fig.savefig(output_dir / "quantization.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
