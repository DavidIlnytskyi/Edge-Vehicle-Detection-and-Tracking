"""Regression tests for archive-to-YOLO dataset normalization."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZIP_DEFLATED, ZipFile

import yaml
from PIL import Image

from src.data import extract_visdrone_archives, repair_data_layout


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (100, 200), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


class ExtractVisDroneArchivesTests(TestCase):
    def test_normalizes_wrapped_and_unwrapped_archives_into_yolo_layout(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            train_archive = temp_dir / "VisDrone2019-DET-train.zip"
            val_archive = temp_dir / "valid.zip"

            # The train archive has the usual enclosing VisDrone directory.
            with ZipFile(train_archive, "w", ZIP_DEFLATED) as archive:
                archive.writestr(
                    "VisDrone2019-DET-train/images/train_01.jpg",
                    _jpeg_bytes(),
                )
                archive.writestr(
                    "VisDrone2019-DET-train/annotations/train_01.txt",
                    "10,20,30,40,1,4,0,0\n",
                )

            # The validation archive has no enclosing directory and already
            # contains YOLO labels.
            with ZipFile(val_archive, "w", ZIP_DEFLATED) as archive:
                archive.writestr("images/val_01.jpg", _jpeg_bytes())
                archive.writestr("labels/val_01.txt", "2 0.1 0.2 0.3 0.4\n")
                archive.writestr("images/val_without_objects.jpg", _jpeg_bytes())

            dataset_dir = extract_visdrone_archives(
                [train_archive, val_archive],
                output_dir=temp_dir / "dataset",
            )
            self.assertEqual(repair_data_layout(dataset_dir), dataset_dir)

            self.assertTrue((dataset_dir / "images/train/train_01.jpg").is_file())
            self.assertTrue((dataset_dir / "images/val/val_01.jpg").is_file())
            self.assertFalse((dataset_dir / "VisDrone2019-DET-train").exists())

            self.assertEqual(
                (dataset_dir / "labels/train/train_01.txt").read_text(encoding="utf-8"),
                "4 0.250000 0.200000 0.300000 0.200000",
            )
            self.assertEqual(
                (dataset_dir / "labels/val/val_01.txt").read_text(encoding="utf-8"),
                "2 0.1 0.2 0.3 0.4\n",
            )
            self.assertEqual(
                (dataset_dir / "labels/val/val_without_objects.txt").read_text(encoding="utf-8"),
                "",
            )

            with (dataset_dir / "data.yaml").open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)

            self.assertEqual(config["path"], str(dataset_dir.resolve()))
            self.assertEqual(config["train"], "images/train")
            self.assertEqual(config["val"], "images/val")
            self.assertNotIn("test", config)
