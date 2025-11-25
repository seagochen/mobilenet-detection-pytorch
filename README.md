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
│       ├── dataset.py        # Dataset classes (YOLO format)
│       ├── transforms.py     # Data augmentation
│       ├── metrics.py        # Evaluation metrics (mAP, etc.)
│       ├── plots.py          # Training visualization
│       ├── callbacks.py      # Training callbacks (EMA, early stopping, etc.)
│       ├── general.py        # General utilities
│       └── visualize.py      # Detection visualization
├── scripts/                   # Training and inference scripts
│   ├── train.py              # Training script
│   └── detect.py             # Inference script
├── runs/                      # Training outputs (created during training)
│   └── train/                # Experiment directories
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
git clone https://github.com/seagochen/mobilenet-detection-pytorch.git
cd mobilenet-detection-pytorch

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

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

### 2. Train the Model

The training script supports many command-line options. All configurations are passed via arguments - no config files needed.

```bash
# List available backbone models
python scripts/train.py --list-backbones

# Basic training
python scripts/train.py --data path/to/dataset.yaml --backbone mobilenetv3_large_100

# Training with custom settings
python scripts/train.py --data path/to/dataset.yaml \
                        --backbone mobilenetv3_small_100 \
                        --img-size 640 \
                        --batch-size 16 \
                        --epochs 100 \
                        --lr 0.001

# Training with advanced features
python scripts/train.py --data path/to/dataset.yaml \
                        --backbone mobilenetv3_large_100 \
                        --amp \               # Mixed precision training
                        --ema \               # Exponential Moving Average
                        --patience 10         # Early stopping
```

Training outputs are saved to `runs/train/exp/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Latest checkpoint
- `metrics.csv` - Training metrics
- Training curves and detection samples

### 3. Run Inference

```bash
# Detect objects in a single image
python scripts/detect.py --weights runs/train/exp/weights/best.pt \
                         --source image.jpg

# Process a directory of images
python scripts/detect.py --weights runs/train/exp/weights/best.pt \
                         --source path/to/images/

# Save detection results
python scripts/detect.py --weights runs/train/exp/weights/best.pt \
                         --source image.jpg \
                         --save-txt
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

### Available Backbones

The training script supports multiple backbone architectures. Use `--list-backbones` to see all options:

```bash
python scripts/train.py --list-backbones
```

Supported backbone families:
- **MobileNetV2**: mobilenetv2_050, mobilenetv2_100, mobilenetv2_140, etc.
- **MobileNetV3-Small**: mobilenetv3_small_050, mobilenetv3_small_100
- **MobileNetV3-Large**: mobilenetv3_large_075, mobilenetv3_large_100 (recommended)
- **MobileNetV4**: mobilenetv4_conv_small, mobilenetv4_conv_medium, mobilenetv4_hybrid_medium
- **EfficientNet-Lite**: efficientnet_lite0, efficientnet_lite1, efficientnet_lite2
- **MNASNet**: mnasnet_050, mnasnet_100

## Export and Deployment

### Export to ONNX

```python
import torch
from mobilenet_ssdlite.models import MobileNetDetector

# Load model
model = MobileNetDetector(config)
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

1. **Start with pretrained backbone**: Enabled by default (use `--no-pretrained` to disable)
2. **Use appropriate batch size**: 16-32 for 640x640 on GPUs with 8-16GB VRAM
3. **Learning rate**: Default 1e-3 with cosine annealing schedule
4. **Mixed precision**: Use `--amp` flag for faster training with less memory
5. **EMA**: Use `--ema` for more stable model weights

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

- [ ] Add model export utilities (ONNX, TensorRT)
- [ ] Mobile deployment examples (iOS/Android)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
