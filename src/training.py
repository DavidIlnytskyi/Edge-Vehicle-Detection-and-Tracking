"""Training helpers built on top of Ultralytics YOLO."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def load_model(weights: str = "yolo11n.pt") -> YOLO:
    return YOLO(weights)


def train_model(
    data_config: str | Path,
    weights: str = "yolo11n.pt",
    epochs: int = 10,
    image_size: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str = "visdrone",
    device: str | int | None = None,
):
    model = load_model(weights)
    return model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=project,
        name=name,
        device=device,
    )


def validate_model(
    data_config: str | Path,
    weights: str = "yolo11n.pt",
    split: str = "val",
    image_size: int = 640,
    batch: int = 16,
    device: str | int | None = None,
):
    model = load_model(weights)
    return model.val(
        data=str(data_config),
        split=split,
        imgsz=image_size,
        batch=batch,
        device=device,
    )

