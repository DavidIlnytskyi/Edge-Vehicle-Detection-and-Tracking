"""Dataset preparation and reporting helpers."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml
import os

from src.constants import CLASS_NAMES
from src.dataset import YoloDetectionDataset
from src.utils import convert_to_yolo_format, load_yolo_label_rows


_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def _infer_split_name(archive_path: Path) -> str:
    """Infer the standard YOLO split name from a VisDrone archive name."""
    name = archive_path.stem.casefold()

    if "train" in name:
        return "train"
    if "val" in name or "valid" in name:
        return "val"
    if "test" in name and "challenge" in name:
        return "test_challenge"
    if "test" in name:
        return "test"

    raise ValueError(
        f"Cannot infer a dataset split from {archive_path.name!r}. "
        "Name the archive with train, val/valid, or test, or rename it before extraction."
    )


def _files_in_named_directories(
    root: Path,
    directory_names: set[str],
    suffixes: set[str] | frozenset[str],
) -> list[Path]:
    """Return files below directories with one of ``directory_names``.

    Looking below the named directory rather than relying on a top-level folder is
    what makes ``dataset/images/...`` and ``images/...`` archive layouts behave
    identically.
    """
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in suffixes:
            continue

        if any(parent.name.casefold() in directory_names for parent in path.parents):
            files.append(path)

    return sorted(files)


def _index_files_by_stem(paths: list[Path], kind: str) -> dict[str, Path]:
    """Build a case-insensitive image/annotation lookup and reject ambiguity."""
    indexed: dict[str, Path] = {}

    for path in paths:
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError(
                f"Duplicate {kind} filename {path.stem!r}: {indexed[key]} and {path}."
            )
        indexed[key] = path

    return indexed


def _build_yolo_split(source_dir: Path, split: str, destination_dir: Path) -> None:
    """Copy one archive into ``images/<split>`` and ``labels/<split>``.

    VisDrone's ``annotations`` are converted to YOLO labels. If an archive
    already contains a ``labels`` directory, those labels are retained instead.
    """
    image_paths = _files_in_named_directories(source_dir, {"images"}, _IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No images directory was found in the {split!r} archive.")

    yolo_label_paths = _files_in_named_directories(source_dir, {"labels"}, {".txt"})
    annotation_paths = _files_in_named_directories(source_dir, {"annotations"}, {".txt"})
    label_paths = yolo_label_paths or annotation_paths
    labels_are_yolo = bool(yolo_label_paths)

    images_by_stem = _index_files_by_stem(image_paths, "image")
    labels_by_stem = _index_files_by_stem(label_paths, "label")
    unmatched_labels = sorted(set(labels_by_stem) - set(images_by_stem))
    if unmatched_labels:
        preview = ", ".join(unmatched_labels[:5])
        raise ValueError(
            f"The {split!r} archive has labels with no matching image: {preview}."
        )

    image_destination = destination_dir / "images" / split
    label_destination = destination_dir / "labels" / split
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    for image_key, image_path in images_by_stem.items():
        target_image = image_destination / image_path.name
        target_label = label_destination / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)

        source_label = labels_by_stem.get(image_key)
        if source_label is None:
            # Empty files make image/label pairs explicit and represent images
            # without objects in the standard YOLO layout.
            target_label.touch()
        elif labels_are_yolo:
            shutil.copy2(source_label, target_label)
        else:
            convert_to_yolo_format(
                source_label,
                image_path=image_path,
                output_path=target_label,
            )


def extract_visdrone_archives(
    archive_paths: list[str | Path],
    output_dir: str | Path = "data",
    data_yaml_path: str | Path | None = None,
) -> Path:
    """Extract VisDrone archives into a standard YOLO dataset.

    The generated layout is independent of whether an archive contains an
    enclosing directory::

        <output_dir>/images/train
        <output_dir>/images/val
        <output_dir>/labels/train
        <output_dir>/labels/val
        <output_dir>/data.yaml

    A ``test`` archive is written to matching ``test`` directories and included
    in ``data.yaml``; the VisDrone test-challenge archive is kept separately as
    ``test_challenge``. Existing requested splits are replaced only after every
    supplied archive has been successfully staged.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_archives = [Path(path) for path in archive_paths]
    splits = [_infer_split_name(path) for path in normalized_archives]
    duplicate_splits = sorted({split for split in splits if splits.count(split) > 1})
    if duplicate_splits:
        raise ValueError(f"Only one archive is allowed per split: {', '.join(duplicate_splits)}.")

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_dataset = temp_dir / "dataset"

        for index, (archive_path, split) in enumerate(zip(normalized_archives, splits)):
            archive_extract_dir = temp_dir / f"archive-{index}"
            archive_extract_dir.mkdir()
            with ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(archive_extract_dir)

            _build_yolo_split(archive_extract_dir, split, staged_dataset)

        for split in splits:
            for kind in ("images", "labels"):
                target_dir = output_dir / kind / split
                staged_dir = staged_dataset / kind / split
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged_dir), str(target_dir))

    yaml_output = Path(data_yaml_path) if data_yaml_path is not None else output_dir / "data.yaml"
    available_splits = {split for split in splits if (output_dir / "images" / split).exists()}
    write_data_yaml(
        output_path=yaml_output,
        # Ultralytics resolves a relative ``path`` from its datasets directory,
        # not from this YAML file. An absolute dataset root works from any CWD.
        dataset_root=output_dir.resolve(),
        train_images="images/train" if "train" in available_splits else None,
        val_images="images/val" if "val" in available_splits else None,
        test_images="images/test" if "test" in available_splits else None,
    )

    return output_dir


def repair_data_layout(data_dir: str | Path, fallback_split: str = "VisDrone2019-DET-test-dev") -> Path:
    """Repair the legacy per-split layout without changing a YOLO dataset."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if images_dir.is_dir() and labels_dir.is_dir() and any(images_dir.iterdir()):
        # ``extract_visdrone_archives`` already produces this layout. Older
        # notebooks call this helper after extraction, so moving these folders
        # into a legacy split would undo the normalization.
        return data_dir

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


def load_split_datasets(data_dir):
    datasets = dict()
    data_yaml = data_dir / "data.yaml"
    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    for split_type in ["val", "test", "train"]:
        datasets[split_type] = YoloDetectionDataset(data_dir, config[split_type].split("/", 1)[1])

    return datasets


def build_dataset_summary(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)

    rows = []

    for split_name in ["train", "test", "val"]:
        images = data_path / "images" / split_name
        labels = data_path / "labels" / split_name

        image_paths = list(images.glob("*.jpg"))
        label_paths = list(labels.glob("*.txt"))

        image_count = len(image_paths)
        labels_count = len(label_paths)

        box_count = sum(
            len(load_yolo_label_rows(path))
            for path in label_paths
        )

        avg_boxes_per_image = (
            round(box_count / image_count, 2)
            if image_count else 0.0
        )

        rows.append({
            "split": split_name,
            "images": image_count,
            "label_files": labels_count,
            "boxes": box_count,
            "avg_boxes_per_image": avg_boxes_per_image,
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values("images", ascending=False)
        .reset_index(drop=True)
    )

    total_images = summary["images"].sum()

    summary["image_share"] = (
        (summary["images"] / total_images).round(2)
        if total_images else 0.0
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
    train_images: str | None = "data/images/train",
    val_images: str | None = "data/images/val",
    test_images: str | None = None,
) -> Path:
    """Write a YOLO data configuration using paths relative to ``dataset_root``."""
    output_path = Path(output_path)
    data = {"path": str(Path(dataset_root))}

    for split, image_dir in (("train", train_images), ("val", val_images), ("test", test_images)):
        if image_dir is not None:
            data[split] = image_dir

    data["names"] = CLASS_NAMES

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return output_path

def create_tiler_layout(src, temp):
    src = Path(src).resolve()
    temp = Path(temp)

    if temp.exists():
        shutil.rmtree(temp)

    split_mapping = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    for src_split, dst_split in split_mapping.items():
        dst_images = temp / dst_split / "images"
        dst_labels = temp / dst_split / "labels"

        dst_images.mkdir(parents=True)
        dst_labels.mkdir(parents=True)

        for image in (src / "images" / src_split).iterdir():
            if image.is_file():
                os.link(image, dst_images / image.name)

        for label in (src / "labels" / src_split).glob("*.txt"):
            os.link(label, dst_labels / label.name)

    shutil.copy2(src / "data.yaml", temp / "data.yaml")

def convert_tiler_to_yolo(root):
    root = Path(root)

    (root / "images").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)

    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    for tiler_split, yolo_split in split_mapping.items():
        split_dir = root / tiler_split

        if not split_dir.exists():
            continue

        shutil.move(
            split_dir / "images",
            root / "images" / yolo_split,
        )

        shutil.move(
            split_dir / "labels",
            root / "labels" / yolo_split,
        )

        split_dir.rmdir()


def fix_data_yaml(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    yaml_path = dataset_dir / "data.yaml"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = str(dataset_dir)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


