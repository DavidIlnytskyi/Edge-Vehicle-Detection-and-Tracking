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


def summarize_split_with_classes(split_dir: str | Path) -> tuple[dict[str, float | int | str], Counter[int]]:
    """Build a per-split summary and class counter."""
    split_dir = Path(split_dir)
    image_count = len(list((split_dir / "images").glob("*.jpg")))
    label_files = sorted((split_dir / "labels").glob("*.txt"))
    class_counts: Counter[int] = Counter({class_id: 0 for class_id in CLASS_NAMES})
    box_count = 0

    for label_path in label_files:
        for class_id, *_ in load_yolo_label_rows(label_path):
            class_counts[int(class_id)] += 1
            box_count += 1

    summary = {
        "split": split_dir.name,
        "images": image_count,
        "label_files": len(label_files),
        "boxes": box_count,
        "avg_boxes_per_image": box_count / image_count if image_count else 0,
    }
    return summary, class_counts


def build_split_summary_and_class_counts(
    data_dir: str | Path,
) -> tuple[list[Path], pd.DataFrame, dict[str, Counter[int]]]:
    """Return split directories, a summary dataframe, and per-split class counts."""
    split_dirs = iter_split_dirs(data_dir)
    split_summaries = []
    class_distributions = {}

    for split_dir in split_dirs:
        summary, counts = summarize_split_with_classes(split_dir)
        split_summaries.append(summary)
        class_distributions[split_dir.name] = counts

    summary_df = pd.DataFrame(split_summaries).sort_values("images", ascending=False)
    total_images = summary_df["images"].sum()
    summary_df["image_share"] = summary_df["images"] / total_images if total_images else 0.0

    return split_dirs, summary_df, class_distributions


def build_class_distribution_frame(class_distributions: dict[str, Counter[int]]) -> pd.DataFrame:
    """Convert per-split class counters to a dataframe sorted by total box count."""
    class_df = pd.DataFrame(class_distributions).rename(index=CLASS_NAMES)
    class_df["total"] = class_df.sum(axis=1)
    return class_df.sort_values("total", ascending=True)


def collect_bbox_metrics(split_dirs: list[Path]) -> pd.DataFrame:
    """Collect width, height, area, and aspect-ratio stats for all boxes."""
    bbox_rows = []

    for split_dir in split_dirs:
        split_name = split_dir.name

        for label_path in sorted((split_dir / "labels").glob("*.txt")):
            image_path = change_file_type(label_path)
            image_width, image_height = get_image_size(image_path)

            for class_id, _, _, norm_width, norm_height in load_yolo_label_rows(label_path):
                width = norm_width * image_width
                height = norm_height * image_height

                bbox_rows.append(
                    {
                        "split": split_name,
                        "class_name": CLASS_NAMES[int(class_id)],
                        "width": width,
                        "height": height,
                        "area": width * height,
                        "aspect_ratio": width / height if height else 0,
                    }
                )

    return pd.DataFrame(bbox_rows)


def collect_bbox_size_data(split_dirs: list[Path]) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Classify boxes into COCO-style size buckets."""
    bbox_size_rows = []

    for split_dir in split_dirs:
        split_name = split_dir.name
        image_dir = split_dir / "images"

        for label_path in sorted((split_dir / "labels").glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.jpg"
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

                bbox_size_rows.append(
                    {
                        "split": split_name,
                        "path": image_path,
                        "bbox_area_px": bbox_area_px,
                        "size_bucket": size_bucket,
                    }
                )

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


def inspect_dataset_issues(data_dir: str | Path) -> list[object]:
    """Run CleanVision inspection for each dataset split."""
    from cleanvision import Imagelab

    inspection_results = []
    for data_file in Path(data_dir).iterdir():
        imagelab = Imagelab(data_path=str(data_file))
        imagelab.find_issues()
        imagelab.report()
        inspection_results.append(imagelab)

    return inspection_results


def remove_issue_files(inspection_results: list[object], logger: logging.Logger | None = None) -> None:
    """Remove files flagged by CleanVision alongside their matching annotation file."""
    logger = logger or logging.getLogger(__name__)

    for image_lab in inspection_results:
        for issue_path, _issue_data in image_lab.issues.iterrows():
            try:
                Path(issue_path).unlink()
                change_file_type(issue_path).unlink()
                logger.debug("%s was removed.", issue_path)
            except Exception:
                logger.debug("Failed to remove %s.", issue_path, exc_info=True)


def find_out_of_boundaries_images(split_dirs: list[Path]) -> list[Path]:
    """Return images whose label rows extend past image boundaries.

    The check mirrors the notebook's original logic to preserve behavior.
    """
    out_of_boundaries_images = []

    for split_dir in split_dirs:
        image_dir = split_dir / "images"

        for label_path in sorted((split_dir / "labels").glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.jpg"
            image_width, image_height = get_image_size(image_path)

            for _, x_0, y_0, width, height in load_yolo_label_rows(label_path):
                if (x_0 + width) > image_width or (y_0 + height) > image_height:
                    out_of_boundaries_images.append(image_path)

    return out_of_boundaries_images


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
