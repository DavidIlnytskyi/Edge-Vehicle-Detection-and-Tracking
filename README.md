### Installation

```
conda create -n image-object-detection python=3.13

conda activate image-object-detection

pip install -r requirements.txt
```

### Project layout

- `src/data.py`: dataset extraction, label conversion, split summaries, and `data.yaml` generation
- `src/visualization.py`: reusable image and bounding-box plotting helpers
- `src/training.py`: Ultralytics YOLO training and validation wrappers
- `data_loading.ipynb`: data preparation workflow
- `data_vizualization.ipynb`: exploratory analysis and sample inspection
- `model_work.ipynb`: model training and validation workflow

### Dataset extraction

`extract_visdrone_archives` accepts both VisDrone archive variants: archives
with an enclosing dataset directory and archives with `images`/`annotations`
directly at the archive root. It normalizes them into YOLO's layout and writes
`data/data.yaml` automatically.

```python
from src.data import extract_visdrone_archives

extract_visdrone_archives([
    "zips/VisDrone2019-DET-train.zip",
    "zips/VisDrone2019-DET-val.zip",
])
```

The resulting training configuration is `data/data.yaml`; the repository-level
`data.yaml` references the same default `data/images/train` and
`data/images/val` directories.
