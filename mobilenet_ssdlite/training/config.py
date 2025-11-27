"""
Training configuration and backbone registry.
Centralized configuration management for MobileNet-YOLO training.
"""
import timm
from typing import Dict, Any, List, Optional


# ============== Available Backbone Registry ==============
AVAILABLE_BACKBONES = {
    # MobileNetV2 series
    'mobilenetv2_050': {'params': '2.0M', 'desc': 'MobileNetV2 0.5x width'},
    'mobilenetv2_100': {'params': '3.5M', 'desc': 'MobileNetV2 1.0x width (standard)'},
    'mobilenetv2_110d': {'params': '4.5M', 'desc': 'MobileNetV2 1.1x width, deeper'},
    'mobilenetv2_120d': {'params': '5.8M', 'desc': 'MobileNetV2 1.2x width, deeper'},
    'mobilenetv2_140': {'params': '6.1M', 'desc': 'MobileNetV2 1.4x width'},

    # MobileNetV3 Small series (lightweight, for edge devices)
    'mobilenetv3_small_050': {'params': '1.0M', 'desc': 'MobileNetV3-Small 0.5x (ultra-light)'},
    'mobilenetv3_small_075': {'params': '1.5M', 'desc': 'MobileNetV3-Small 0.75x'},
    'mobilenetv3_small_100': {'params': '2.5M', 'desc': 'MobileNetV3-Small 1.0x'},

    # MobileNetV3 Large series (better performance)
    'mobilenetv3_large_075': {'params': '4.0M', 'desc': 'MobileNetV3-Large 0.75x'},
    'mobilenetv3_large_100': {'params': '5.5M', 'desc': 'MobileNetV3-Large 1.0x (recommended)'},

    # MobileNetV4 series (latest generation)
    'mobilenetv4_conv_small': {'params': '3.8M', 'desc': 'MobileNetV4-Conv Small'},
    'mobilenetv4_conv_medium': {'params': '9.7M', 'desc': 'MobileNetV4-Conv Medium'},
    'mobilenetv4_conv_large': {'params': '32.6M', 'desc': 'MobileNetV4-Conv Large'},
    'mobilenetv4_hybrid_medium': {'params': '11.1M', 'desc': 'MobileNetV4-Hybrid Medium (with attention)'},
    'mobilenetv4_hybrid_large': {'params': '37.8M', 'desc': 'MobileNetV4-Hybrid Large (with attention)'},

    # EfficientNet-Lite series
    'efficientnet_lite0': {'params': '4.7M', 'desc': 'EfficientNet-Lite0 (good accuracy/speed)'},
    'efficientnet_lite1': {'params': '5.4M', 'desc': 'EfficientNet-Lite1'},
    'efficientnet_lite2': {'params': '6.1M', 'desc': 'EfficientNet-Lite2'},

    # MNASNet series
    'mnasnet_050': {'params': '1.0M', 'desc': 'MNASNet 0.5x'},
    'mnasnet_100': {'params': '4.4M', 'desc': 'MNASNet 1.0x'},

    # ResNet series (standard conv, for debugging)
    'resnet18': {'params': '11.7M', 'desc': 'ResNet-18 (standard conv, for debugging)'},
    'resnet34': {'params': '21.8M', 'desc': 'ResNet-34 (standard conv, for debugging)'},
    'resnet50': {'params': '25.6M', 'desc': 'ResNet-50 (standard conv, for debugging)'},
}

# ============== Default Anchor Configuration ==============
DEFAULT_ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],       # Small objects (stride 8)
    [[30, 61], [62, 45], [59, 119]],      # Medium objects (stride 16)
    [[116, 90], [156, 198], [373, 326]]   # Large objects (stride 32)
]


def list_available_backbones() -> None:
    """Print all available backbones in a formatted table."""
    print("\n" + "=" * 70)
    print("Available MobileNet Backbones for Object Detection")
    print("=" * 70)

    categories = {
        'MobileNetV2': [k for k in AVAILABLE_BACKBONES if k.startswith('mobilenetv2')],
        'MobileNetV3-Small': [k for k in AVAILABLE_BACKBONES if 'v3_small' in k],
        'MobileNetV3-Large': [k for k in AVAILABLE_BACKBONES if 'v3_large' in k],
        'MobileNetV4': [k for k in AVAILABLE_BACKBONES if k.startswith('mobilenetv4')],
        'EfficientNet-Lite': [k for k in AVAILABLE_BACKBONES if k.startswith('efficientnet_lite')],
        'MNASNet': [k for k in AVAILABLE_BACKBONES if k.startswith('mnasnet')],
        'ResNet (Debug)': [k for k in AVAILABLE_BACKBONES if k.startswith('resnet')],
    }

    for category, backbones in categories.items():
        if backbones:
            print(f"\n{category}:")
            print("-" * 50)
            for name in backbones:
                info = AVAILABLE_BACKBONES[name]
                print(f"  {name:<30} {info['params']:<8} {info['desc']}")

    print("\n" + "=" * 70)
    print("Usage: python train.py --data coco.yaml --backbone mobilenetv3_large_100")
    print("=" * 70 + "\n")


def validate_backbone(name: str) -> bool:
    """
    Validate if a backbone name is valid.

    Args:
        name: Backbone model name

    Returns:
        True if valid, False otherwise
    """
    if name in AVAILABLE_BACKBONES:
        return True
    # Try to load from timm
    try:
        model = timm.create_model(name, pretrained=False, features_only=True, out_indices=(2, 3, 4))
        del model
        return True
    except Exception:
        return False


def build_config(args) -> Dict[str, Any]:
    """
    Build configuration dictionary from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration dictionary
    """
    config = {
        'model': {
            'backbone': args.backbone,
            'pretrained': not args.no_pretrained,
            'num_classes': 80,  # Will be overridden by dataset
            'input_size': [args.img_size, args.img_size],
            'fpn_channels': args.fpn_channels,
            'num_anchors': 3
        },
        'anchors': DEFAULT_ANCHORS,
        'train': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.workers,
            'optimizer': {
                'type': 'AdamW',
                'lr': args.lr,
                'weight_decay': args.weight_decay
            },
            'loss_weights': {
                'box': args.lambda_box,
                'obj': args.lambda_obj,
                'cls': args.lambda_cls
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5
            }
        },
        'val': {
            'batch_size': args.batch_size * 2,
            'conf_thresh': args.conf_thres,
            'iou_thresh': args.iou_thres
        },
        'data': {
            'yaml': args.data
        }
    }
    return config


def get_backbone_info(name: str) -> Optional[Dict[str, str]]:
    """
    Get information about a backbone.

    Args:
        name: Backbone name

    Returns:
        Dictionary with 'params' and 'desc' keys, or None if not found
    """
    return AVAILABLE_BACKBONES.get(name)


def get_all_backbone_names() -> List[str]:
    """Get list of all registered backbone names."""
    return list(AVAILABLE_BACKBONES.keys())
