"""Dataset preparation and reporting helpers."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

from src.constants import CLASS_NAMES
from src.dataset import YoloDetectionDataset
from src.utils import convert_to_yolo_format, load_yolo_label_rows


def _move_extracted_contents(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    entries = [path for path in source_dir.iterdir() if path.name != "__MACOSX"]

    if len(entries) == 1 and entries[0].is_dir():
        nested_dir = entries[0]
        if nested_dir.name == destination_dir.name or not (nested_dir / "images").exists():
            entries = [path for path in nested_dir.iterdir() if path.name != "__MACOSX"]

    for entry in entries:
        target_path = destination_dir / entry.name
        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        shutil.move(str(entry), str(target_path))


def extract_visdrone_archives(
    archive_paths: list[str | Path],
    output_dir: str | Path = "data",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for archive_path in archive_paths:
        archive_path = Path(archive_path)
        split_dir = output_dir / archive_path.stem

        with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            with ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            _move_extracted_contents(temp_dir, split_dir)

    return output_dir


def repair_data_layout(data_dir: str | Path, fallback_split: str = "VisDrone2019-DET-test-dev") -> Path:
    data_dir = Path(data_dir)
    fallback_dir = data_dir / fallback_split
    misplaced_names = ("images", "annotations", "labels")

    misplaced_entries = [data_dir / name for name in misplaced_names if (data_dir / name).exists()]
    if not misplaced_entries:
        return data_dir

    fallback_dir.mkdir(parents=True, exist_ok=True)

    for entry in misplaced_entries:
        target_path = fallback_dir / entry.name
        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        shutil.move(str(entry), str(target_path))

    return data_dir


def iter_split_dirs(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(path for path in data_dir.iterdir() if path.is_dir())


def convert_split_annotations(split_dir: str | Path) -> int:
    split_dir = Path(split_dir)
    annotation_dir = split_dir / "annotations"
    converted = 0

    for annotation_path in sorted(annotation_dir.glob("*.txt")):
        convert_to_yolo_format(annotation_path)
        converted += 1

    return converted


def prepare_all_splits(data_dir: str | Path) -> dict[str, int]:
    stats = {}
    for split_dir in iter_split_dirs(data_dir):
        stats[split_dir.name] = convert_split_annotations(split_dir)

    return stats


def load_split_datasets(data_dir: str | Path) -> dict[str, YoloDetectionDataset]:
    datasets = {}

    for split_dir in iter_split_dirs(data_dir):
        label_dir = split_dir / "labels"
        if not label_dir.exists():
            continue

        datasets[split_dir.name] = YoloDetectionDataset(split_dir)

    return datasets


def summarize_split(split_dir: str | Path) -> dict[str, float | int | str]:
    split_dir = Path(split_dir)
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    image_count = len(list(image_dir.glob("*.jpg")))
    label_paths = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    box_count = sum(len(load_yolo_label_rows(path)) for path in label_paths)

    return {
        "split": split_dir.name,
        "images": image_count,
        "label_files": len(label_paths),
        "boxes": box_count,
        "avg_boxes_per_image": round(box_count / image_count, 2) if image_count else 0.0,
    }


def build_dataset_summary(data_dir: str | Path) -> pd.DataFrame:
    rows = [summarize_split(split_dir) for split_dir in iter_split_dirs(data_dir)]
    summary = pd.DataFrame(rows).sort_values("images", ascending=False).reset_index(drop=True)

    total_images = summary["images"].sum()
    summary["image_share"] = (
        (summary["images"] / total_images).round(2) if total_images else 0.0
    )

    return summary


def build_class_distribution(data_dir: str | Path) -> pd.DataFrame:
    counts: Counter[int] = Counter()

    for split_dir in iter_split_dirs(data_dir):
        label_dir = split_dir / "labels"
        if not label_dir.exists():
            continue

        for label_path in label_dir.glob("*.txt"):
            for class_id, *_ in load_yolo_label_rows(label_path):
                counts[int(class_id)] += 1

    rows = [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "boxes": counts.get(class_id, 0),
        }
        for class_id in sorted(CLASS_NAMES)
    ]
    distribution = pd.DataFrame(rows).sort_values("boxes", ascending=False).reset_index(drop=True)

    total_boxes = distribution["boxes"].sum()
    distribution["box_share"] = (
        (distribution["boxes"] / total_boxes).round(4) if total_boxes else 0.0
    )

    return distribution


def write_data_yaml(
    output_path: str | Path = "data.yaml",
    dataset_root: str | Path = ".",
    train_images: str = "data/VisDrone2019-DET-train/images",
    val_images: str = "data/VisDrone2019-DET-val/images",
    test_images: str = "data/VisDrone2019-DET-test-dev/images",
) -> Path:
    output_path = Path(output_path)
    data = {
        "path": str(Path(dataset_root)),
        "train": train_images,
        "val": val_images,
        "test": test_images,
        "names": CLASS_NAMES,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return output_path
