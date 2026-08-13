from pathlib import Path
from pathlib import Path
from PIL import Image

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
            str(file_path.with_suffix(".txt")).replace("images", "annotations")
        )

    return file_path

def convert_to_yolo_format(txt_path: Path):
    output_dir = txt_path.parent.parent / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = change_file_type(txt_path)

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

    output_path = output_dir / txt_path.name

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(converted_lines))