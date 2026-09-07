"""Analysis helpers for the data visualization notebook."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from src.constants import CLASS_NAMES
from src.data import iter_split_dirs
from src.utils import change_file_type, load_yolo_label_rows


def get_image_size(image_path: str | Path) -> tuple[int, int]:
    """Return image dimensions as ``(width, height)``."""
    with Image.open(image_path) as image:
        return image.size


def summarize_split_with_classes(
    data_path: str | Path,
    split_name: str,
) -> tuple[dict[str, float | int | str], Counter[int]]:

    data_path = Path(data_path)

    images_dir = data_path / "images" / split_name
    labels_dir = data_path / "labels" / split_name

    image_paths = list(images_dir.glob("*.jpg"))
    label_paths = list(labels_dir.glob("*.txt"))

    image_count = len(image_paths)

    class_counts: Counter[int] = Counter({
        class_id: 0 for class_id in CLASS_NAMES
    })

    box_count = 0

    for label_path in label_paths:
        for class_id, *_ in load_yolo_label_rows(label_path):
            class_counts[int(class_id)] += 1
            box_count += 1

    summary = {
        "split": split_name,
        "images": image_count,
        "label_files": len(label_paths),
        "boxes": box_count,
        "avg_boxes_per_image": (
            round(box_count / image_count, 2)
            if image_count else 0.0
        ),
    }

    return summary, class_counts


def build_split_summary_and_class_counts(
    data_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Counter[int]]]:

    split_summaries = []
    class_distributions = {}

    for split_name in ["train", "test", "val"]:
        summary, counts = summarize_split_with_classes(
            data_path,
            split_name,
        )

        split_summaries.append(summary)
        class_distributions[split_name] = counts

    summary_df = (
        pd.DataFrame(split_summaries)
        .sort_values("images", ascending=False)
        .reset_index(drop=True)
    )

    total_images = summary_df["images"].sum()

    summary_df["image_share"] = (
        (summary_df["images"] / total_images).round(2)
        if total_images else 0.0
    )

    return summary_df, class_distributions

def build_class_distribution_frame(class_distributions: dict[str, Counter[int]]) -> pd.DataFrame:
    """Convert per-split class counters to a dataframe sorted by total box count."""
    class_df = pd.DataFrame(class_distributions).rename(index=CLASS_NAMES)
    class_df["total"] = class_df.sum(axis=1)
    return class_df.sort_values("total", ascending=True)

def collect_bbox_metrics(
    data_path: str | Path,
) -> pd.DataFrame:

    data_path = Path(data_path)

    bbox_rows = []

    for split_name in ["train", "test", "val"]:
        images_dir = data_path / "images" / split_name
        labels_dir = data_path / "labels" / split_name

        label_paths = sorted(labels_dir.glob("*.txt"))

        for label_path in label_paths:
            image_path = images_dir / f"{label_path.stem}.jpg"

            image_width, image_height = get_image_size(image_path)

            for class_id, _, _, norm_width, norm_height in load_yolo_label_rows(label_path):
                width = norm_width * image_width
                height = norm_height * image_height

                bbox_rows.append({
                    "split": split_name,
                    "class_name": CLASS_NAMES[int(class_id)],
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "aspect_ratio": width / height if height else 0.0,
                })

    return pd.DataFrame(bbox_rows)

def collect_bbox_size_data(
    data_path: str | Path,
) -> tuple[list[dict[str, object]], pd.DataFrame]:

    data_path = Path(data_path)

    bbox_size_rows = []

    for split_name in ["train", "test", "val"]:
        images_dir = data_path / "images" / split_name
        labels_dir = data_path / "labels" / split_name

        label_paths = sorted(labels_dir.glob("*.txt"))

        for label_path in label_paths:
            image_path = images_dir / f"{label_path.stem}.jpg"

            image_width, image_height = get_image_size(image_path)

            for _, _, _, norm_width, norm_height in load_yolo_label_rows(label_path):
                width = norm_width * image_width
                height = norm_height * image_height

                bbox_area_px = width * height

                if bbox_area_px < 32**2:
                    size_bucket = "small"
                elif bbox_area_px < 96**2:
                    size_bucket = "medium"
                else:
                    size_bucket = "large"

                bbox_size_rows.append({
                    "split": split_name,
                    "path": image_path,
                    "bbox_area_px": bbox_area_px,
                    "size_bucket": size_bucket,
                })

    return bbox_size_rows, pd.DataFrame(bbox_size_rows)


def summarize_bbox_size_distribution(
    bbox_size_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and per-split box-size percentages."""
    size_distribution = (
        bbox_size_df["size_bucket"]
        .value_counts(normalize=True)
        .reindex(["small", "medium", "large"], fill_value=0)
        .mul(100)
        .rename("percentage")
        .to_frame()
    )

    size_distribution_by_split = (
        bbox_size_df.groupby("split")["size_bucket"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percentage")
        .reset_index()
        .pivot(index="split", columns="size_bucket", values="percentage")
        .reindex(columns=["small", "medium", "large"], fill_value=0)
    )

    return size_distribution, size_distribution_by_split


def inspect_dataset_issues(
    data_path: str | Path,
) -> list[object]:

    from cleanvision import Imagelab

    data_path = Path(data_path)

    inspection_results = []

    for split_name in ["train", "test", "val"]:
        images_dir = data_path / "images" / split_name

        imagelab = Imagelab(data_path=str(images_dir))

        imagelab.find_issues()
        imagelab.report()

        inspection_results.append(imagelab)

    return inspection_results


def remove_issue_files(inspection_results: list[object], logger: logging.Logger | None = None) -> None:
    """Remove files flagged by CleanVision alongside their matching annotation file."""
    for image_lab in inspection_results:
        for issue_path, _issue_data in image_lab.issues.iterrows():
            try:
                Path(issue_path).unlink()
                change_file_type(issue_path).unlink()
                logger.debug("%s was removed.", issue_path)
            except Exception:
                logger.debug("Failed to remove %s.", issue_path, exc_info=True)


def collect_small_images(
    bbox_size_rows: list[dict[str, object]],
    split_name: str = "VisDrone2019-DET-train",
) -> list[Path]:
    """Return paths for images that contain small boxes in the selected split."""
    small_images = []
    for row in bbox_size_rows:
        if row["split"] == split_name and row["size_bucket"] == "small":
            small_images.append(row["path"])

    return small_images
