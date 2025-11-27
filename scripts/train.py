"""
Training script for MobileNet-YOLO Object Detection.

This is the main entry point for training. All training logic has been
modularized into the mobilenet_ssdlite.training package.

Usage:
    python train.py --data path/to/dataset.yaml --backbone mobilenetv3_large_100
    python train.py --list-backbones  # Show available backbones
"""
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mobilenet_ssdlite.training import (
    list_available_backbones,
    validate_backbone,
    build_config,
    Trainer,
)
from mobilenet_ssdlite.utils import init_seeds, colorstr, increment_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MobileNet-YOLO Object Detection')

    # ===== Backbone selection =====
    parser.add_argument('--list-backbones', action='store_true',
                        help='List all available backbones and exit')
    parser.add_argument('--backbone', type=str, default='mobilenetv3_large_100',
                        help='Backbone model name (default: mobilenetv3_large_100)')

    # ===== Data =====
    parser.add_argument('--data', type=str, required=False, default='',
                        help='Dataset YAML file path (required for training)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers (default: 8)')

    # ===== Model =====
    parser.add_argument('--fpn-channels', type=int, default=128,
                        help='FPN output channels (default: 128)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained backbone weights')

    # ===== Training =====
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help='Weight decay (default: 0.00005)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping max norm (default: 10.0)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='LR scheduler: cosine or plateau (default: cosine)')

    # ===== Loss weights =====
    parser.add_argument('--lambda-box', type=float, default=7.5,
                        help='Box loss weight (default: 7.5)')
    parser.add_argument('--lambda-obj', type=float, default=1.0,
                        help='Objectness loss weight (default: 1.0)')
    parser.add_argument('--lambda-cls', type=float, default=0.5,
                        help='Classification loss weight (default: 0.5)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (default: 0.0)')

    # ===== Advanced features =====
    parser.add_argument('--ema', action='store_true',
                        help='Use Exponential Moving Average')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='Factor to reduce learning rate (default: 0.1)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping after N LR reductions (0=disabled)')

    # ===== Evaluation =====
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluate every N epochs (default: 1)')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='Confidence threshold for evaluation')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                        help='IoU threshold for NMS')
    parser.add_argument('--plot-samples', type=int, default=16,
                        help='Number of detection samples to plot')

    # ===== Output =====
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Save directory (default: runs/train)')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name (default: exp)')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from experiment name')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (default: auto)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')

    # ===== Anchor =====
    parser.add_argument('--auto-anchor', action='store_true', default=True,
                        help='Auto compute anchors from dataset')
    parser.add_argument('--no-auto-anchor', action='store_true',
                        help='Disable auto anchor computation')

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Handle --list-backbones
    if args.list_backbones:
        list_available_backbones()
        return

    # Check required arguments
    if not args.data:
        print(colorstr('bright_red', "Error: --data is required"))
        print("Usage: python train.py --data path/to/dataset.yaml")
        return

    # Validate backbone
    if not validate_backbone(args.backbone):
        print(colorstr('bright_red', f"Error: Unknown backbone '{args.backbone}'"))
        print("Use --list-backbones to see available options")
        return

    # Initialize
    init_seeds(args.seed)

    # Setup directories
    if args.resume:
        save_dir = Path(args.project) / args.resume
        if not save_dir.exists():
            print(colorstr('bright_red', f'Experiment not found: {save_dir}'))
            return
    else:
        save_dir = Path(increment_path(Path(args.project) / args.name))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = build_config(args)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')

    # Print info
    print(colorstr('bright_green', f'\nStarting training on {device}'))
    print(colorstr('bright_cyan', f'Backbone: {args.backbone}'))
    print(colorstr('bright_cyan', f'Image size: {args.img_size}x{args.img_size}'))
    print(colorstr('bright_cyan', f'FPN channels: {args.fpn_channels}'))
    if args.label_smoothing > 0:
        print(colorstr('bright_cyan', f'Label smoothing: {args.label_smoothing}'))

    # Create trainer and run
    trainer = Trainer(config, args, save_dir, device)
    trainer.setup_data()
    trainer.setup_anchors()
    trainer.setup_model()

    if args.resume:
        if not trainer.resume():
            return

    trainer.train()


if __name__ == '__main__':
    main()
