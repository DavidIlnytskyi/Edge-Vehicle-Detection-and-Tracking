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
