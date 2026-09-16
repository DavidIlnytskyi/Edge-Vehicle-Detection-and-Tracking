"""YOLO-aware image augmentation utilities for road-scene object detection.

The transforms in this module run *before* image tiling.  They therefore keep
the source image dimensions and update YOLO bounding boxes as part of every
geometric operation.  Tile the exported dataset afterwards so small objects
remain intact until the tiler creates its training crops.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class AugmentationPolicy:
    """A named, documented Albumentations policy ready for YOLO boxes."""

    name: str
    purpose: str
    transform: A.Compose


@dataclass(frozen=True)
class ExportSummary:
    """Counts returned after generating an augmented split."""

    source_images: int
    copied_originals: int
    augmented_images: int
    input_boxes: int
    output_boxes: int


def _bbox_params() -> A.BboxParams:
    """Return one consistent box policy for every augmentation preset.

    ``min_visibility`` only affects boxes clipped by the small translation in
    ``Affine``.  ``min_area`` remains zero because VisDrone contains genuinely
    tiny objects that become useful after the subsequent tiling step.
    """

    return A.BboxParams(
        coord_format="yolo",
        label_fields=["class_labels"],
        min_area=0.0,
        min_visibility=0.20,
    )


def _compose(transforms: Sequence[A.BasicTransform], seed: int | None) -> A.Compose:
    return A.Compose(list(transforms), bbox_params=_bbox_params(), seed=seed)


def _road_scene_geometry() -> list[A.BasicTransform]:
    """Mild geometry that is plausible for forward-facing traffic cameras."""

    return [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            rotate=(0.0, 0.0),
            shear=(0.0, 0.0),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.25,
        ),
    ]


def make_augmentation_policies(seed: int | None = 42) -> dict[str, AugmentationPolicy]:
    """Build comparable augmentation policies for traffic-object detection.

    Start training with ``baseline``.  Treat every other policy as an ablation:
    compare it against the baseline on the same validation split before folding
    it into ``strong``.  Each policy uses a deterministic internal random
    stream when ``seed`` is supplied, making notebook comparisons repeatable.
    """

    policies = {
        "baseline": AugmentationPolicy(
            name="baseline",
            purpose="Mild left/right viewpoint, exposure, and compression variation for normal training.",
            transform=_compose(
                [
                    *_road_scene_geometry(),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.15, 0.15),
                        contrast_limit=(-0.15, 0.15),
                        p=0.35,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=12,
                        val_shift_limit=12,
                        p=0.20,
                    ),
                    A.OneOf(
                        [
                            A.ImageCompression(quality_range=(65, 95), p=1.0),
                            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                            A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                        ],
                        p=0.20,
                    ),
                ],
                seed,
            ),
        ),
        "lighting": AugmentationPolicy(
            name="lighting",
            purpose="Dawn, dusk, backlight, and automatic-exposure robustness.",
            transform=_compose(
                [
                    *_road_scene_geometry(),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.35, 0.20),
                                contrast_limit=(-0.25, 0.25),
                                p=1.0,
                            ),
                            A.RandomGamma(gamma_limit=(65, 135), p=1.0),
                            A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
                        ],
                        p=0.70,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=18,
                        val_shift_limit=18,
                        p=0.35,
                    ),
                ],
                None if seed is None else seed + 1,
            ),
        ),
        "degradation": AugmentationPolicy(
            name="degradation",
            purpose="Motion, focus, sensor-noise, and lossy-stream robustness.",
            transform=_compose(
                [
                    *_road_scene_geometry(),
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=(3, 7), p=1.0),
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
                            A.ImageCompression(quality_range=(35, 80), p=1.0),
                        ],
                        p=0.55,
                    ),
                ],
                None if seed is None else seed + 2,
            ),
        ),
        "weather": AugmentationPolicy(
            name="weather",
            purpose="Weather and cast-shadow robustness; enable only if it matches deployment data.",
            transform=_compose(
                [
                    *_road_scene_geometry(),
                    A.OneOf(
                        [
                            A.RandomFog(p=1.0),
                            A.RandomRain(p=1.0),
                            A.RandomShadow(p=1.0),
                        ],
                        p=0.35,
                    ),
                ],
                None if seed is None else seed + 3,
            ),
        ),
        "occlusion": AugmentationPolicy(
            name="occlusion",
            purpose="Partial-object robustness for traffic, poles, and overlapping vehicles.",
            transform=_compose(
                [
                    *_road_scene_geometry(),

                    A.OneOf(
                        [
                            A.ConstrainedCoarseDropout(
                                num_holes_range=(1, 2),
                                hole_height_range=(0.2, 0.5),
                                hole_width_range=(0.2, 0.5),
                                bbox_labels=[
                                    "pedestrian",
                                    "people",
                                    "bicycle",
                                    "car",
                                    "van",
                                    "truck",
                                    "tricycle",
                                    "awning_tricycle",
                                    "bus",
                                    "motor",
                                ],
                                fill="random_uniform",
                                p=1.0,
                            ),

                            A.CoarseDropout(
                                num_holes_range=(1, 3),
                                hole_height_range=(0.03, 0.12),
                                hole_width_range=(0.03, 0.12),
                                fill="random_uniform",
                                p=1.0,
                            ),
                        ],
                        p=0.35,
                    ),
                ],
                None if seed is None else seed + 4,
            ),
        ),
        "strong": AugmentationPolicy(
            name="strong",
            purpose="A deliberately harder mixed policy; use only after individual ablations validate it.",
            transform=_compose(
                [
                    *_road_scene_geometry(),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.30, 0.20),
                                contrast_limit=(-0.20, 0.20),
                                p=1.0,
                            ),
                            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                            A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
                        ],
                        p=0.55,
                    ),
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=(3, 7), p=1.0),
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
                            A.ImageCompression(quality_range=(35, 85), p=1.0),
                        ],
                        p=0.45,
                    ),
                    A.OneOf([A.RandomFog(p=1.0), A.RandomRain(p=1.0), A.RandomShadow(p=1.0)], p=0.15),
                    A.CoarseDropout(
                        num_holes_range=(1, 2),
                        hole_height_range=(0.03, 0.08),
                        hole_width_range=(0.03, 0.08),
                        fill="random_uniform",
                        p=0.15,
                    ),
                ],
                None if seed is None else seed + 5,
            ),
        ),
    }
    return policies


def yolo_label_path(image_path: str | Path) -> Path:
    """Infer ``labels/<split>/<stem>.txt`` from ``images/<split>/<image>``."""

    image_path = Path(image_path)
    if image_path.parent.parent.name != "images":
        raise ValueError(
            "Cannot infer a YOLO label path. Expected an image under "
            "<dataset>/images/<split>/, or pass label_path explicitly."
        )
    return image_path.parent.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def read_yolo_annotations(label_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO text file as ``(boxes[N, 4], class_labels[N])`` arrays."""

    label_path = Path(label_path)
    boxes: list[list[float]] = []
    class_labels: list[int] = []

    with label_path.open(encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = stripped.split()
            if len(values) != 5:
                raise ValueError(
                    f"{label_path}:{line_number} must have 5 YOLO values; found {len(values)}."
                )

            class_id, *box = map(float, values)
            if not class_id.is_integer() or class_id < 0:
                raise ValueError(f"{label_path}:{line_number} has an invalid class id: {class_id}.")
            if not all(np.isfinite(box)) or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{label_path}:{line_number} has an invalid bounding box: {box}.")
            if any(value < 0 or value > 1 for value in box):
                raise ValueError(f"{label_path}:{line_number} has a non-normalized bounding box: {box}.")

            class_labels.append(int(class_id))
            boxes.append(box)

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(class_labels, dtype=np.int64),
    )


def load_yolo_sample(
    image_path: str | Path,
    label_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an RGB image and its normalized YOLO boxes and classes."""

    image_path = Path(image_path)
    label_path = Path(label_path) if label_path is not None else yolo_label_path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image was not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"YOLO label file was not found: {label_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    boxes, class_labels = read_yolo_annotations(label_path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), boxes, class_labels


def apply_augmentation(
    image: np.ndarray,
    boxes: np.ndarray,
    class_labels: np.ndarray,
    policy: AugmentationPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a policy to an image and YOLO labels without changing inputs."""

    result = policy.transform(
        image=image,
        bboxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4).tolist(),
        class_labels=np.asarray(class_labels, dtype=np.int64).reshape(-1).tolist(),
    )
    transformed_boxes = np.asarray(result["bboxes"], dtype=np.float32).reshape(-1, 4)
    transformed_labels = np.asarray(result["class_labels"], dtype=np.int64).reshape(-1)
    return result["image"], transformed_boxes, transformed_labels


def _draw_yolo_boxes(
    axis,
    image: np.ndarray,
    boxes: np.ndarray,
    class_labels: np.ndarray,
    class_names: Mapping[int, str] | None = None,
) -> None:
    """Draw normalized YOLO boxes over an RGB image."""

    height, width = image.shape[:2]
    axis.imshow(image)
    for box, class_id in zip(boxes, class_labels):
        x_center, y_center, box_width, box_height = box
        x_min = (x_center - box_width / 2) * width
        y_min = (y_center - box_height / 2) * height
        rectangle = Rectangle(
            (x_min, y_min),
            box_width * width,
            box_height * height,
            fill=False,
            linewidth=1.2,
            edgecolor=plt.cm.tab20(int(class_id) % 20),
        )
        axis.add_patch(rectangle)
        if class_names:
            axis.text(
                x_min,
                max(0, y_min - 2),
                class_names.get(int(class_id), str(class_id)),
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
    axis.axis("off")


def plot_augmentation_comparison(
    reference_path: str | Path,
    policies: Mapping[str, AugmentationPolicy],
    policy_names: Iterable[str] | None = None,
    *,
    label_path: str | Path | None = None,
    class_names: Mapping[int, str] | None = None,
    columns: int = 3,
):
    """Show original and selected policy outputs from one ``reference_path``.

    The reference image is read once and each output is drawn with the boxes
    returned by Albumentations, so box movement or filtering is immediately
    visible before any dataset is exported.
    """

    image, boxes, class_labels = load_yolo_sample(reference_path, label_path)
    selected_names = list(policy_names) if policy_names is not None else list(policies)
    missing = [name for name in selected_names if name not in policies]
    if missing:
        raise KeyError(f"Unknown augmentation policies: {missing}")
    if columns < 1:
        raise ValueError("columns must be at least 1.")

    panel_count = len(selected_names) + 1
    rows = int(np.ceil(panel_count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    flattened_axes = axes.ravel()

    _draw_yolo_boxes(flattened_axes[0], image, boxes, class_labels, class_names)
    flattened_axes[0].set_title(f"original\nboxes: {len(boxes)}")

    for axis, policy_name in zip(flattened_axes[1:], selected_names):
        policy = policies[policy_name]
        augmented_image, augmented_boxes, augmented_labels = apply_augmentation(
            image, boxes, class_labels, policy
        )
        _draw_yolo_boxes(axis, augmented_image, augmented_boxes, augmented_labels, class_names)
        axis.set_title(f"{policy.name}\nboxes: {len(boxes)} → {len(augmented_boxes)}")

    for axis in flattened_axes[panel_count:]:
        axis.axis("off")
    figure.tight_layout()
    return figure


def write_yolo_annotations(
    label_path: str | Path,
    boxes: np.ndarray,
    class_labels: np.ndarray,
) -> Path:
    """Write validated normalized YOLO boxes, including intentionally empty labels."""

    label_path = Path(label_path)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    class_labels = np.asarray(class_labels, dtype=np.int64).reshape(-1)
    if len(boxes) != len(class_labels):
        raise ValueError("boxes and class_labels must contain the same number of rows.")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as label_file:
        for class_id, (x_center, y_center, width, height) in zip(class_labels, boxes):
            label_file.write(
                f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )
    return label_path


def _image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _prepare_output_split(output_root: Path, split: str) -> tuple[Path, Path]:
    output_image_dir = output_root / "images" / split
    output_label_dir = output_root / "labels" / split
    if any(directory.exists() and any(directory.iterdir()) for directory in (output_image_dir, output_label_dir)):
        raise FileExistsError(
            f"Refusing to overwrite an existing output split at {output_root}. "
            "Choose a new output directory or remove that split deliberately."
        )
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    return output_image_dir, output_label_dir


def copy_yolo_split(
    dataset_root: str | Path,
    output_root: str | Path,
    split: str,
) -> int:
    """Copy an unaugmented split, normally validation, without overwriting it."""

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    source_image_dir = dataset_root / "images" / split
    source_label_dir = dataset_root / "labels" / split
    if not source_image_dir.is_dir() or not source_label_dir.is_dir():
        raise FileNotFoundError(f"Missing images or labels for split {split!r} in {dataset_root}.")
    output_image_dir, output_label_dir = _prepare_output_split(output_root, split)

    image_paths = _image_paths(source_image_dir)
    for image_path in image_paths:
        label_path = source_label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        shutil.copy2(image_path, output_image_dir / image_path.name)
        shutil.copy2(label_path, output_label_dir / label_path.name)
    return len(image_paths)


def export_augmented_split(
    dataset_root: str | Path,
    output_root: str | Path,
    policy: AugmentationPolicy,
    *,
    split: str = "train",
    copies_per_image: int = 1,
    include_originals: bool = True,
    jpeg_quality: int = 95,
) -> ExportSummary:
    """Create YOLO-ready augmented copies of one split before tiling.

    The function never changes ``dataset_root`` and refuses to write into an
    existing output split.  Set ``include_originals=True`` to make the later
    tiler see both the original and augmented source images.
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    if dataset_root.resolve() == output_root.resolve():
        raise ValueError("output_root must be different from dataset_root.")
    if copies_per_image < 1:
        raise ValueError("copies_per_image must be at least 1.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")

    source_image_dir = dataset_root / "images" / split
    source_label_dir = dataset_root / "labels" / split
    if not source_image_dir.is_dir() or not source_label_dir.is_dir():
        raise FileNotFoundError(f"Missing images or labels for split {split!r} in {dataset_root}.")
    output_image_dir, output_label_dir = _prepare_output_split(output_root, split)

    image_paths = _image_paths(source_image_dir)
    input_boxes = 0
    output_boxes = 0
    copied_originals = 0
    augmented_images = 0

    for image_path in image_paths:
        label_path = source_label_dir / f"{image_path.stem}.txt"
        image, boxes, class_labels = load_yolo_sample(image_path, label_path)
        input_boxes += len(boxes)

        if include_originals:
            shutil.copy2(image_path, output_image_dir / image_path.name)
            shutil.copy2(label_path, output_label_dir / label_path.name)
            copied_originals += 1

        for copy_index in range(copies_per_image):
            augmented_image, augmented_boxes, augmented_labels = apply_augmentation(
                image, boxes, class_labels, policy
            )
            output_name = f"{image_path.stem}__{policy.name}_{copy_index + 1:02d}{image_path.suffix}"
            output_image_path = output_image_dir / output_name
            image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            image_write_parameters = (
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                if output_image_path.suffix.lower() in {".jpg", ".jpeg"}
                else []
            )
            write_success = cv2.imwrite(str(output_image_path), image_bgr, image_write_parameters)
            if not write_success:
                raise OSError(f"OpenCV could not write augmented image: {output_image_path}")
            write_yolo_annotations(output_label_dir / f"{output_image_path.stem}.txt", augmented_boxes, augmented_labels)
            output_boxes += len(augmented_boxes)
            augmented_images += 1

    return ExportSummary(
        source_images=len(image_paths),
        copied_originals=copied_originals,
        augmented_images=augmented_images,
        input_boxes=input_boxes,
        output_boxes=output_boxes,
    )


def write_yolo_data_yaml(
    dataset_root: str | Path,
    *,
    output_path: str | Path | None = None,
    train_split: str = "train",
    val_split: str = "val",
    class_names: Mapping[int, str] | None = None,
) -> Path:
    """Write a portable Ultralytics dataset configuration for an exported split."""

    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path) if output_path is not None else dataset_root / "data.yaml"
    data = {
        "path": str(dataset_root),
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
    }
    if class_names is not None:
        data["names"] = dict(class_names)
    with output_path.open("w", encoding="utf-8") as data_file:
        yaml.safe_dump(data, data_file, sort_keys=False)
    return output_path
