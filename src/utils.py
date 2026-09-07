from pathlib import Path

from PIL import Image

import torch
import torch.nn.functional as F

import numpy as np
from collections import Counter, defaultdict


def yolo_to_xyxy_pixels(box, img_w, img_h):
    x_center, y_center, width, height = box

    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]

def xywhn_to_xyxy(boxes, h, w):
    """
    YOLO normalized [x_center, y_center, width, height]
    -> pixel [x1, y1, x2, y2]
    """
    boxes = boxes.clone()

    x = boxes[:, 0] * w
    y = boxes[:, 1] * h
    bw = boxes[:, 2] * w
    bh = boxes[:, 3] * h

    boxes[:, 0] = x - bw / 2
    boxes[:, 1] = y - bh / 2
    boxes[:, 2] = x + bw / 2
    boxes[:, 3] = y + bh / 2

    return boxes

def xyxy_to_yolo(box, image_width, image_height):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return np.array([
        cx / image_width,
        cy / image_height,
        w / image_width,
        h / image_height,
    ])

def box_iou_batch(box1, box2):
    """
    box1: [N, 4]
    box2: [M, 4]
    returns IoU matrix [N, M]
    """
    if len(box1) == 0 or len(box2) == 0:
        return torch.zeros((len(box1), len(box2)))

    top_left = torch.maximum(box1[:, None, :2], box2[None, :, :2])
    bottom_right = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])

    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = (
        (box1[:, 2] - box1[:, 0]) *
        (box1[:, 3] - box1[:, 1])
    )

    area2 = (
        (box2[:, 2] - box2[:, 0]) *
        (box2[:, 3] - box2[:, 1])
    )

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / union.clamp(min=1e-6)


def load_yolo_labels(label_path, h, w):
    """
    Returns:
        gt_boxes: [N, 4] xyxy
        gt_classes: [N]
    """
    labels = []

    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append([float(x) for x in line.split()])
    except FileNotFoundError:
        labels = []

    if not labels:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long)

    labels = torch.tensor(labels)

    gt_classes = labels[:, 0].long()
    gt_boxes = xywhn_to_xyxy(labels[:, 1:5], h, w)

    return gt_boxes, gt_classes

def box_iou_raw(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def get_image_size(image_path):
    with Image.open(image_path) as image:
        return image.width, image.height


def analyze_missed_objects(
    file_paths,
    saved_preds,
    iou_threshold=0.5,
    conf_threshold=0.25,
):
    missed = []
    class_stats = Counter()
    size_stats = Counter()
    total_boxes = 0

    for idx, (image_path, label_path) in enumerate(file_paths):
        gt_objects = []

        w, h = get_image_size(image_path)

        with open(label_path) as f:
            for line in f:
                values = list(map(float, line.split()))
                cls = int(values[0])
                box = values[1:5]

                gt_objects.append((cls, box))

        total_boxes += len(gt_objects)

        pred_boxes = saved_preds[idx][0]
        pred_classes = saved_preds[idx][1]
        pred_conf = saved_preds[idx][2]

        predictions = [
            (cls, box)
            for cls, box, conf in zip(
                pred_classes,
                pred_boxes,
                pred_conf,
            )
            if conf >= conf_threshold
        ]

        for gt_cls, gt_box in gt_objects:
            best_iou = 0

            for pred_cls, pred_box in predictions:
                if pred_cls != gt_cls:
                    continue

                gt_box_xyxy = yolo_to_xyxy_pixels(gt_box, w, h)

                iou = box_iou_raw(pred_box, gt_box_xyxy)

                best_iou = max(best_iou, iou)

                area = 1

                if best_iou < iou_threshold:
                    gx, gy, gw, gh = gt_box
                    area = gw * gh

                if area < 0.01:
                    size = "tiny"
                elif area < 0.04:
                    size = "small"
                elif area < 0.16:
                    size = "medium"
                else:
                    size = "large"

                missed.append({
                    "image": image_path,
                    "class": gt_cls,
                    "box": gt_box,
                    "best_iou": best_iou,
                    "area": area,
                    "size": size,
                })

                class_stats[gt_cls] += 1
                size_stats[size] += 1

    return missed, class_stats, size_stats, total_boxes

def yolo_preprocess(image: torch.Tensor, size: int = 640) -> torch.Tensor:
    image = image.float()

    if image.max() > 1:
        image = image / 255.0

    _, h, w = image.shape

    scale = min(size / h, size / w)

    new_h = round(h * scale)
    new_w = round(w * scale)

    image = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_h = size - new_h
    pad_w = size - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = F.pad(
        image,
        (left, right, top, bottom),
        value=114 / 255.0,
    )

    return image

def change_file_type(file_path: str | Path) -> Path:
    file_path = Path(file_path)

    if file_path.suffix == ".txt":
        # annotations/foo.txt -> images/foo.jpg
        file_path = Path(
            str(file_path.with_suffix(".jpg")).replace("annotations", "images")
        )
        file_path = Path(str(file_path).replace("labels", "images"))
    else:
        # images/foo.jpg -> annotations/foo.txt
        file_path = Path(
            str(file_path.with_suffix(".txt")).replace("images", "labels")
        )

    return file_path


def load_yolo_label_rows(label_path: str | Path) -> list[tuple[float, float, float, float, float]]:
    label_path = Path(label_path)
    rows = []

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            class_id, x_center, y_center, width, height = map(float, stripped.split())
            rows.append((class_id, x_center, y_center, width, height))

    return rows


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    box_width = width * image_width
    box_height = height * image_height
    x_min = (x_center * image_width) - box_width / 2
    y_min = (y_center * image_height) - box_height / 2

    return x_min, y_min, box_width, box_height


def convert_to_yolo_format(
    txt_path: str | Path,
    image_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Convert one VisDrone annotation file into its corresponding YOLO label.

    ``image_path`` and ``output_path`` are optional for backward compatibility
    with the original VisDrone ``annotations``/``images`` directory layout.
    """
    txt_path = Path(txt_path)
    if output_path is None:
        output_dir = txt_path.parent.parent / "labels"
        output_path = output_dir / txt_path.name
    else:
        output_path = Path(output_path)

    if image_path is None:
        image_path = change_file_type(txt_path)
    image_path = Path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    converted_lines = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lst = line.split(",")

            x_min = float(lst[0])
            y_min = float(lst[1])
            width = float(lst[2])
            height = float(lst[3])
            class_id = lst[5]

            # Convert top-left + width/height -> center + width/height
            x_center = x_min + width / 2
            y_center = y_min + height / 2

            # Normalize to [0, 1]
            x_center /= image_width
            y_center /= image_height
            width /= image_width
            height /= image_height

            converted_line = (
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            converted_lines.append(converted_line)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(converted_lines))

    return output_path
