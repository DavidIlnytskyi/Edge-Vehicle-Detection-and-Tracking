"""Tests for class-aware recall evaluation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

from src.recall import calculate_recall, evaluate_recall


class RecallEvaluationTests(TestCase):
    def test_matches_each_ground_truth_once_and_requires_a_matching_class(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            image_path = temp_dir / "image.jpg"
            label_path = temp_dir / "image.txt"
            Image.new("RGB", (100, 100)).save(image_path)
            label_path.write_text(
                "4 0.5 0.5 0.2 0.2\n4 0.5 0.5 0.2 0.2\n5 0.2 0.2 0.2 0.2\n",
                encoding="utf-8",
            )

            summary = evaluate_recall(
                [(image_path, label_path)],
                [(
                    np.array([[40, 40, 60, 60], [10, 10, 30, 30]]),
                    np.array([4, 4]),
                    np.array([0.95, 0.40]),
                )],
                conf_threshold=0.5,
                iou_threshold=0.5,
            )

            self.assertEqual(summary.ground_truth_count, 3)
            self.assertEqual(summary.true_positives, 1)
            self.assertEqual(summary.false_negatives, 2)
            self.assertAlmostEqual(summary.recall, 1 / 3)

    def test_legacy_box_confidence_results_remain_supported(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            image_path = temp_dir / "image.jpg"
            label_path = temp_dir / "image.txt"
            Image.new("RGB", (100, 100)).save(image_path)
            label_path.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            recall = calculate_recall(
                [(image_path, label_path)],
                [(np.array([[40, 40, 60, 60]]), np.array([0.9]))],
            )

            self.assertEqual(recall, 1.0)

    def test_rejects_a_result_count_that_does_not_match_the_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number"):
            evaluate_recall([], [None])
