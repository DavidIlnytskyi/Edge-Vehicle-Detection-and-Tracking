"""Utilities for evaluating class-aware YOLO recall."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from src.utils import box_iou_batch, get_image_size, load_yolo_labels


ImageLabelPair: TypeAlias = tuple[str | Path, str | Path]
Prediction: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
LegacyPrediction: TypeAlias = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class RecallSummary:
    """Counts and recall from one-to-one prediction-to-label matching."""

    true_positives: int
    ground_truth_count: int

    @property
    def false_negatives(self) -> int:
        """Number of ground-truth objects that were not detected."""
        return self.ground_truth_count - self.true_positives

    @property
    def recall(self) -> float:
        """Return recall, using zero for a set with no ground-truth boxes."""
        if self.ground_truth_count == 0:
            return 0.0
        return self.true_positives / self.ground_truth_count


def collect_yolo_predictions(
    model: Callable[..., Sequence[Any]],
    image_paths: Sequence[str | Path],
    *,
    confidence: float = 0.25,
    image_size: int = 640,
) -> list[Prediction]:
    """Run a YOLO model and return ``(boxes, classes, confidences)`` per image.

    Boxes use pixel ``xyxy`` coordinates.  The model is deliberately duck-typed
    so that callers may pass an ``ultralytics.YOLO`` instance without making
    Ultralytics an import-time dependency of this module.
    """
    _validate_threshold("confidence", confidence)

    predictions: list[Prediction] = []
    for image_path in image_paths:
        result = model(
            str(image_path),
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            predictions.append(_empty_prediction())
            continue

        predictions.append((
            _as_numpy(boxes.xyxy, dtype=np.float32),
            _as_numpy(boxes.cls, dtype=np.int64),
            _as_numpy(boxes.conf, dtype=np.float32),
        ))

    return predictions


def evaluate_recall(
    image_paths: Sequence[ImageLabelPair],
    results: Sequence[Prediction | LegacyPrediction | None],
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> RecallSummary:
    """Evaluate detections against YOLO labels using one-to-one matching.

    A prediction must meet both thresholds and, when its class is supplied,
    have the same class as its matched ground-truth box.  A two-value legacy
    prediction of ``(boxes, confidences)`` is also accepted and skips class
    matching for compatibility with earlier notebook output.
    """
    _validate_threshold("conf_threshold", conf_threshold)
    _validate_threshold("iou_threshold", iou_threshold)

    if len(image_paths) != len(results):
        raise ValueError(
            "image_paths and results must contain the same number of items."
        )

    total_ground_truth = 0
    true_positives = 0

    for (image_path, label_path), result in zip(image_paths, results):
        image_width, image_height = get_image_size(image_path)
        gt_boxes, gt_classes = load_yolo_labels(
            label_path,
            image_height,
            image_width,
        )
        total_ground_truth += len(gt_boxes)

        if len(gt_boxes) == 0 or result is None:
            continue

        pred_boxes, pred_classes, pred_confidences = _prepare_prediction(result)
        eligible_prediction_indexes = torch.nonzero(
            pred_confidences >= conf_threshold,
            as_tuple=False,
        ).flatten()

        if len(eligible_prediction_indexes) == 0:
            continue

        ious = box_iou_batch(pred_boxes, gt_boxes)
        ordered_indexes = eligible_prediction_indexes[
            torch.argsort(pred_confidences[eligible_prediction_indexes], descending=True)
        ]
        matched_ground_truth: set[int] = set()

        for prediction_index in ordered_indexes.tolist():
            valid_ground_truth = [
                ground_truth_index
                for ground_truth_index in range(len(gt_boxes))
                if ground_truth_index not in matched_ground_truth
                and (
                    pred_classes is None
                    or gt_classes[ground_truth_index].item()
                    == pred_classes[prediction_index].item()
                )
            ]

            if not valid_ground_truth:
                continue

            candidate_ious = ious[prediction_index, valid_ground_truth]
            best_position = torch.argmax(candidate_ious).item()
            best_ground_truth = valid_ground_truth[best_position]

            if candidate_ious[best_position].item() >= iou_threshold:
                matched_ground_truth.add(best_ground_truth)
                true_positives += 1

    return RecallSummary(
        true_positives=true_positives,
        ground_truth_count=total_ground_truth,
    )


def calculate_recall(
    image_paths: Sequence[ImageLabelPair],
    results: Sequence[Prediction | LegacyPrediction | None],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> float:
    """Return recall only; use :func:`evaluate_recall` when counts are needed."""
    return evaluate_recall(
        image_paths,
        results,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    ).recall


def _prepare_prediction(
    result: Prediction | LegacyPrediction,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if len(result) not in {2, 3}:
        raise ValueError(
            "Each result must be (boxes, confidences) or "
            "(boxes, classes, confidences)."
        )

    pred_boxes = _as_boxes(result[0])

    if len(result) == 2:
        pred_classes = None
        pred_confidences = _as_vector(result[1], "confidences", torch.float32)
    else:
        pred_classes = _as_vector(result[1], "classes", torch.long)
        pred_confidences = _as_vector(result[2], "confidences", torch.float32)

    if len(pred_boxes) != len(pred_confidences):
        raise ValueError("Each box must have one confidence value.")
    if pred_classes is not None and len(pred_boxes) != len(pred_classes):
        raise ValueError("Each box must have one class value.")

    return pred_boxes, pred_classes, pred_confidences


def _as_boxes(value: Any) -> torch.Tensor:
    boxes = _as_tensor(value, torch.float32)
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Prediction boxes must have shape (N, 4) in xyxy format.")
    return boxes


def _as_vector(value: Any, name: str, dtype: torch.dtype) -> torch.Tensor:
    vector = _as_tensor(value, dtype)
    if vector.ndim > 1:
        raise ValueError(f"Prediction {name} must be one-dimensional.")
    return vector.reshape(-1)


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _as_numpy(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _empty_prediction() -> Prediction:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float32),
    )


def _validate_threshold(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, inclusive.")
