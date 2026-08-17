"""Visualization helpers for YOLO-style detection labels."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.constants import CLASS_NAMES
from src.utils import load_yolo_label_rows, yolo_to_xyxy


def plot_annotated_image(
    image_path: str | Path,
    label_path: str | Path | None = None,
    ax=None,
    class_names: dict[int, str] | None = None,
):
    image_path = Path(image_path)
    label_path = Path(label_path) if label_path else image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    class_names = class_names or CLASS_NAMES

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size
        ax.imshow(image)

    if label_path.exists():
        for class_id, x_center, y_center, width, height in load_yolo_label_rows(label_path):
            class_id = int(class_id)
            x_min, y_min, box_width, box_height = yolo_to_xyxy(
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                image_width=image_width,
                image_height=image_height,
            )

            rectangle = patches.Rectangle(
                (x_min, y_min),
                box_width,
                box_height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rectangle)
            ax.text(
                x_min,
                max(y_min - 4, 0),
                class_names.get(class_id, str(class_id)),
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.7, "pad": 2},
            )

    ax.set_title(image_path.name)
    ax.axis("off")
    return ax


def visualize_random_samples(
    split_dir: str | Path,
    class_ids: list[int] | None = None,
    sample_size: int = 8,
    grid_shape: tuple[int, int] = (2, 4),
):
    split_dir = Path(split_dir)
    label_dir = split_dir / "labels"
    image_dir = split_dir / "images"
    selected_classes = set(CLASS_NAMES if class_ids is None else class_ids)

    candidates = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        rows = load_yolo_label_rows(label_path)
        if any(int(class_id) in selected_classes for class_id, *_ in rows):
            candidates.append(image_path)

    if not candidates:
        raise ValueError(f"No samples found for classes {sorted(selected_classes)} in {split_dir}.")

    sample_count = min(sample_size, len(candidates))
    sampled_images = random.sample(candidates, sample_count)

    fig, axes = plt.subplots(*grid_shape, figsize=(20, 10))
    axes = axes.flatten()

    for axis, image_path in zip(axes, sampled_images):
        plot_annotated_image(image_path, ax=axis)

    for axis in axes[sample_count:]:
        axis.axis("off")

    fig.tight_layout()
    return fig

