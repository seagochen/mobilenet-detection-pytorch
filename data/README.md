# Dataset Directory

This directory is for storing your datasets.

## Using YOLO Format Dataset

### Option 1: Dataset in this directory

Place your dataset here and update the `dataset.yaml` configuration:

```
data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

In `dataset.yaml`, set:
```yaml
path: .  # Current directory
train: images/train
val: images/val
```

### Option 2: Dataset in external directory

Keep your dataset elsewhere and update `dataset.yaml` to point to it:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
```

### Option 3: Multiple datasets

Create multiple yaml config files for different datasets:

```
data/
├── coco.yaml
├── voc.yaml
└── custom.yaml
```

## Example: COCO Dataset

```yaml
path: /data/coco
train: images/train2017
val: images/val2017
nc: 80
names: [person, bicycle, car, ...]
```

## Example: Custom Dataset

```yaml
path: /data/my_dataset
train: images/train
val: images/val
nc: 3
names: [cat, dog, bird]
```

Then train with:
```bash
python scripts/train.py --config configs/yolo_format.yaml
```

Make sure to update `configs/yolo_format.yaml` to point to your dataset yaml file:
```yaml
data:
  yaml: "data/custom.yaml"
```
