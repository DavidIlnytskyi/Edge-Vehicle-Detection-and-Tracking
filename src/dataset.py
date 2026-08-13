from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch


class CustomImageDataset(Dataset):
    def __init__(self, data_folder: Path, transform=None):
        self.data_folder = Path(data_folder)
        self.image_dir = self.data_folder / "images"
        self.label_dir = self.data_folder / "labels"

        self.images = sorted(self.image_dir.glob("*.jpg"))
        self.transform = transform

        # Verify image -> annotation correspondence
        missing_labels = [
            img.name
            for img in self.images
            if not (self.label_dir / f"{img.stem}.txt").exists()
        ]

        if missing_labels:
            raise RuntimeError(
                f"{len(missing_labels)} images have no annotation file. "
                f"Example: {missing_labels[:5]}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        image = decode_image(str(img_path))

        label_path = self.label_dir / f"{img_path.stem}.txt"

        labels = []

        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # YOLO:
                # class_id x_center y_center width height
                values = [float(x) for x in line.split()]
                labels.append(values)

        if labels:
            target = torch.tensor(labels, dtype=torch.float32)
        else:
            target = torch.empty((0, 5), dtype=torch.float32)

        # For detection, transform image + boxes together
        if self.transform:
            image, target = self.transform(image, target)

        return image, target