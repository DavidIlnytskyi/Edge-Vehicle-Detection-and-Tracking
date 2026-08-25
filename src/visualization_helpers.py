"""Plot builders for the data visualization notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def visualize_yolo_prediction(model, image, conf=0.25, imgsz=640):
    results = model(image, conf=conf, imgsz=imgsz)

    annotated = results[0].plot()

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated[..., ::-1])
    plt.axis("off")
    plt.show()

    return results[0]

def plot_split_summary(summary_df: pd.DataFrame):
    """Plot image and bounding-box counts by split."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_specs = [
        ("images", "Images per Split", "Images", ["#355C7D", "#6C5B7B", "#C06C84"]),
        ("boxes", "Bounding Boxes per Split", "Boxes", ["#F67280", "#F8B195", "#99B898"]),
    ]

    for axis, (column, title, ylabel, colors) in zip(axes, plot_specs):
        axis.bar(summary_df["split"], summary_df[column], color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return fig, axes


def plot_class_distribution(class_df: pd.DataFrame):
    """Plot total class counts and split heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].barh(class_df.index, class_df["total"], color="#355C7D")
    axes[0].set_title("Total Boxes by Class")
    axes[0].set_xlabel("Boxes")

    heatmap_values = class_df.drop(columns="total")
    image = axes[1].imshow(heatmap_values, aspect="auto", cmap="Blues")
    axes[1].set_title("Class Distribution by Split")
    axes[1].set_xticks(range(len(heatmap_values.columns)))
    axes[1].set_xticklabels(heatmap_values.columns, rotation=20, ha="right")
    axes[1].set_yticks(range(len(heatmap_values.index)))
    axes[1].set_yticklabels(heatmap_values.index)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Boxes")

    fig.tight_layout()
    return fig, axes


def plot_bbox_distributions(bbox_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Plot width/height/area histograms and split-wise area boxplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    histogram_specs = [
        ((0, 0), "width", "Bounding Box Width Distribution", "Normalized width", "#355C7D"),
        ((0, 1), "height", "Bounding Box Height Distribution", "Normalized height", "#C06C84"),
        ((1, 0), "area", "Bounding Box Area Distribution", "Normalized area", "#6C5B7B"),
    ]

    for (row_idx, col_idx), column, title, xlabel, color in histogram_specs:
        axis = axes[row_idx, col_idx]
        axis.hist(bbox_df[column], bins=40, color=color, alpha=0.9)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")

    axes[1, 0].set_yscale("log")

    split_order = summary_df.sort_values("images", ascending=False)["split"].tolist()
    boxplot_data = [bbox_df.loc[bbox_df["split"] == split_name, "area"] for split_name in split_order]
    axes[1, 1].boxplot(boxplot_data, tick_labels=split_order, vert=True)
    axes[1, 1].set_title("Bounding Box Area by Split")
    axes[1, 1].set_xlabel("Split")
    axes[1, 1].set_ylabel("Normalized area")
    axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.suptitle("Bounding Box Size Distribution", fontsize=14)
    fig.tight_layout()
    return fig, axes
