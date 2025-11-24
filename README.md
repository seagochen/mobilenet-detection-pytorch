# MobileNet-YOLO Object Detection

A lightweight object detection model combining MobileNet backbone (from timm) with YOLO-style detection head.

## Features

- **Lightweight Architecture**: Uses MobileNetV3 from timm as backbone for efficient inference
- **YOLO-Style Detection**: Anchor-based detection with multi-scale predictions
- **Feature Pyramid Network**: FPN for better multi-scale feature fusion
- **Easy Training**: Simple training pipeline with built-in data augmentation
- **Flexible Configuration**: YAML-based configuration system

## Architecture

```
Input Image (640x640)
    ↓
MobileNet Backbone (timm)
    ├── Scale 1 (stride 8)  → 80x80 features
    ├── Scale 2 (stride 16) → 40x40 features
    └── Scale 3 (stride 32) → 20x20 features
    ↓
Feature Pyramid Network
    ↓
Detection Heads (3 scales)
    ├── Box Regression (x, y, w, h)
    ├── Objectness Score
    └── Class Probabilities
    ↓
Post-processing (NMS)
    ↓
Final Detections
```

## Project Structure

```
mobilenet-detection/
├── configs/                    # Configuration files
│   ├── default.yaml           # Legacy format config
│   ├── yolo_format.yaml       # YOLO format config
│   └── mobilenet_small.yaml
├── mobilenet_ssdlite/         # Main package
│   ├── __init__.py
│   ├── models/                # Model architecture
│   │   ├── __init__.py
│   │   ├── mobilenet_yolo.py # Main model
│   │   ├── backbone.py       # MobileNet backbone
│   │   ├── detection_head.py # Detection head
│   │   └── loss.py           # Loss functions
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── dataset.py        # Dataset classes (YOLO & COCO)
│       ├── transforms.py     # Data augmentation
│       └── visualize.py      # Visualization tools
├── scripts/                   # Training and inference scripts
│   ├── train.py              # Training script
│   └── detect.py             # Inference script
├── examples/                  # Usage examples
│   └── basic_usage.py        # Basic usage example
├── data/                      # Dataset directory
│   └── dataset.yaml          # Example dataset config
├── checkpoints/               # Model checkpoints (created during training)
├── logs/                      # Training logs (created during training)
└── setup.py                   # Package setup
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mobilenet-detection.git
cd mobilenet-detection

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 0. Basic Usage Example

To quickly test the model and understand the API:

```bash
python examples/basic_usage.py
```

This will:
- Load the model configuration
- Create a MobileNet-YOLO model
- Run inference on a dummy image
- Display model information

### 1. Prepare Your Dataset

#### Option 1: YOLO Format (Recommended)

Organize your dataset in the YOLO format:

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img3.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   └── val/
│       ├── img3.txt
│       └── ...
└── dataset.yaml  # Dataset configuration
```

Create a `dataset.yaml` file describing your dataset:

```yaml
# Dataset root directory
path: /path/to/your_dataset

# Dataset splits
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 3

# Class names
names:
  - class1
  - class2
  - class3
```

Each label file (`.txt`) contains one bounding box per line in the format:
```
class_id x_center y_center width height
```
where all coordinates are normalized to [0, 1].

#### Option 2: Legacy Format

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── annotations.json  # COCO format (optional)
└── val/
    ├── images/
    └── annotations.json
```

### 2. Configure Your Model

#### For YOLO Format Dataset

Use `configs/yolo_format.yaml` or create your own:

```yaml
model:
  backbone: "mobilenetv3_large_100"  # or mobilenetv3_small_100
  num_classes: 80  # Will be auto-detected from dataset.yaml
  input_size: [640, 640]

data:
  yaml: "path/to/your/dataset.yaml"  # Point to your YOLO dataset config
```

#### For Legacy Format

Edit `configs/default.yaml`:

```yaml
model:
  backbone: "mobilenetv3_large_100"
  num_classes: 80
  input_size: [640, 640]

data:
  train: "data/train"
  val: "data/val"
  names: ["person", "car", "dog"]  # Your class names
```

### 3. Train the Model

```bash
# Train with YOLO format dataset
python scripts/train.py --config configs/yolo_format.yaml --device cuda

# Train with legacy format
python scripts/train.py --config configs/default.yaml --device cuda

# Resume from checkpoint
python scripts/train.py --config configs/yolo_format.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

Training outputs:
- Checkpoints saved to `checkpoints/`
- TensorBoard logs saved to `logs/`
- Best model saved as `checkpoints/best.pth`

### 4. Run Inference

```bash
# Detect objects in a single image
python scripts/detect.py --config configs/default.yaml \
                         --checkpoint checkpoints/best.pth \
                         --source image.jpg \
                         --output output/

# Process a directory of images
python scripts/detect.py --config configs/default.yaml \
                         --checkpoint checkpoints/best.pth \
                         --source data/test/ \
                         --output output/

# Process a video
python scripts/detect.py --config configs/default.yaml \
                         --checkpoint checkpoints/best.pth \
                         --source video.mp4 \
                         --output output/

# Save detection results as text files
python scripts/detect.py --checkpoint checkpoints/best.pth \
                          --source image.jpg \
                          --save-txt
```

### 5. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

## Model Architecture Details

### Backbone (MobileNet from timm)

The model uses MobileNetV3 from the `timm` library as the feature extractor:

- **MobileNetV3-Large**: Better accuracy, ~5.4M parameters
- **MobileNetV3-Small**: Faster inference, ~2.5M parameters

Features are extracted at three scales (stride 8, 16, 32) for multi-scale detection.

### Detection Head

Each detection head predicts:
- **Bounding Box**: 4 values (tx, ty, tw, th) - offsets from anchors
- **Objectness**: 1 value - probability of object presence
- **Class Scores**: N values - class probabilities (N = num_classes)

Total outputs per anchor: 5 + num_classes

### Anchors

The model uses predefined anchors (similar to YOLOv3) at three scales:

```python
# Small objects (stride 8)
[[10, 13], [16, 30], [33, 23]]

# Medium objects (stride 16)
[[30, 61], [62, 45], [59, 119]]

# Large objects (stride 32)
[[116, 90], [156, 198], [373, 326]]
```

You can customize these for your dataset.

## Loss Functions

The model uses a combined loss function:

1. **Box Loss**: MSE loss on bounding box offsets
2. **Objectness Loss**: BCE loss on object presence
3. **Classification Loss**: BCE loss on class probabilities

```python
total_loss = λ_box * box_loss + λ_obj * obj_loss + λ_cls * cls_loss
```

Default weights: `λ_box=0.05, λ_obj=1.0, λ_cls=0.5`

## Performance Tips

### For Better Accuracy:
- Use `mobilenetv3_large_100` backbone
- Increase input size to 800x800 (slower)
- Train for more epochs (300+)
- Use stronger augmentation
- Fine-tune anchor sizes for your dataset

### For Faster Inference:
- Use `mobilenetv3_small_100` backbone
- Reduce input size to 416x416
- Reduce FPN channels to 128
- Use lower confidence threshold

## Customization

### Using Different Backbones

The model supports any timm backbone with `features_only=True`:

```yaml
model:
  backbone: "mobilenetv3_small_100"  # Faster
  # backbone: "efficientnet_b0"      # More accurate
  # backbone: "resnet50"              # Standard choice
```

### Custom Dataset

1. Create COCO-format annotations or organize images in directories
2. Update `configs/default.yaml`:
   ```yaml
   data:
     train: "path/to/train"
     val: "path/to/val"
     names: ["class1", "class2", "class3"]

   model:
     num_classes: 3  # Match number of classes
   ```

### Anchor Customization

To compute optimal anchors for your dataset:

```python
# TODO: Add anchor clustering script
python scripts/compute_anchors.py --data data/train --num-anchors 9
```

## Export and Deployment

### Export to ONNX

```python
import torch
from mobilenet_ssdlite.models import MobileNetYOLO

# Load model
model = MobileNetYOLO(config)
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    'mobilenet_yolo.onnx',
    opset_version=11,
    input_names=['images'],
    output_names=['boxes', 'scores', 'labels']
)
```

### TorchScript

```python
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('mobilenet_yolo.pt')
```

## Model Zoo

| Backbone | Input Size | mAP@0.5 | Params | FPS (V100) | Weights |
|----------|-----------|---------|--------|------------|---------|
| MobileNetV3-Small | 416 | TBD | 2.5M | ~120 | Coming soon |
| MobileNetV3-Large | 640 | TBD | 5.4M | ~80 | Coming soon |

## Training Tips

1. **Start with pretrained backbone**: Set `pretrained: true` in config
2. **Use appropriate batch size**: 16-32 for 640x640 on GPUs with 8-16GB VRAM
3. **Learning rate**: Start with 1e-3, use cosine annealing
4. **Data augmentation**: Enable for training, disable for validation
5. **Monitor training**: Use TensorBoard to track losses

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce input size
- Use gradient accumulation

### Low Accuracy
- Check anchor sizes match your objects
- Increase training epochs
- Use stronger data augmentation
- Verify annotation quality

### Slow Training
- Use mixed precision training (AMP)
- Reduce number of workers if CPU-bound
- Use smaller backbone

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mobilenet_yolo,
  title = {MobileNet-YOLO: Lightweight Object Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/mobilenet-detection}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [YOLOv3](https://arxiv.org/abs/1804.02767) - Detection architecture inspiration
- [MobileNetV3](https://arxiv.org/abs/1905.02244) - Efficient backbone

## TODO

- [ ] Add anchor clustering tool
- [ ] Add evaluation metrics (mAP calculation)
- [ ] Add model export utilities (ONNX, TensorRT)
- [ ] Add mixed precision training
- [ ] Add Mosaic and MixUp augmentation
- [ ] Pre-trained weights on COCO
- [ ] Mobile deployment examples (iOS/Android)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
