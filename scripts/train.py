"""
Training script for MobileNet-YOLO (Enhanced Version)
All configurations via command line arguments - no config files needed
"""
import os
import sys
import argparse
import yaml
import math
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import timm

# 添加项目根目录到path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mobilenet_ssdlite.models import MobileNetDetector
from mobilenet_ssdlite.models.loss import YOLOLoss
from mobilenet_ssdlite.utils import DetectionDataset, YOLODataset, collate_fn, get_transforms

# 引入从 YOLOv2 移植过来的工具类
from mobilenet_ssdlite.utils.metrics import DetectionMetrics
from mobilenet_ssdlite.utils.plots import TrainingPlotter, plot_detection_samples, plot_labels_distribution
from mobilenet_ssdlite.utils.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelEMA, GradientAccumulator, WarmupScheduler
)
from mobilenet_ssdlite.utils.general import init_seeds, colorstr, increment_path, check_img_size
from mobilenet_ssdlite.utils.anchors import get_or_compute_anchors


# ============== 可用的 Backbone 列表 ==============
AVAILABLE_BACKBONES = {
    # MobileNetV2 系列
    'mobilenetv2_050': {'params': '2.0M', 'desc': 'MobileNetV2 0.5x width'},
    'mobilenetv2_100': {'params': '3.5M', 'desc': 'MobileNetV2 1.0x width (standard)'},
    'mobilenetv2_110d': {'params': '4.5M', 'desc': 'MobileNetV2 1.1x width, deeper'},
    'mobilenetv2_120d': {'params': '5.8M', 'desc': 'MobileNetV2 1.2x width, deeper'},
    'mobilenetv2_140': {'params': '6.1M', 'desc': 'MobileNetV2 1.4x width'},

    # MobileNetV3 Small 系列 (轻量级，适合边缘设备)
    'mobilenetv3_small_050': {'params': '1.0M', 'desc': 'MobileNetV3-Small 0.5x (ultra-light)'},
    'mobilenetv3_small_075': {'params': '1.5M', 'desc': 'MobileNetV3-Small 0.75x'},
    'mobilenetv3_small_100': {'params': '2.5M', 'desc': 'MobileNetV3-Small 1.0x'},

    # MobileNetV3 Large 系列 (性能更好)
    'mobilenetv3_large_075': {'params': '4.0M', 'desc': 'MobileNetV3-Large 0.75x'},
    'mobilenetv3_large_100': {'params': '5.5M', 'desc': 'MobileNetV3-Large 1.0x (recommended)'},

    # MobileNetV4 系列 (最新一代)
    'mobilenetv4_conv_small': {'params': '3.8M', 'desc': 'MobileNetV4-Conv Small'},
    'mobilenetv4_conv_medium': {'params': '9.7M', 'desc': 'MobileNetV4-Conv Medium'},
    'mobilenetv4_conv_large': {'params': '32.6M', 'desc': 'MobileNetV4-Conv Large'},
    'mobilenetv4_hybrid_medium': {'params': '11.1M', 'desc': 'MobileNetV4-Hybrid Medium (with attention)'},
    'mobilenetv4_hybrid_large': {'params': '37.8M', 'desc': 'MobileNetV4-Hybrid Large (with attention)'},

    # EfficientNet-Lite 系列
    'efficientnet_lite0': {'params': '4.7M', 'desc': 'EfficientNet-Lite0 (good accuracy/speed)'},
    'efficientnet_lite1': {'params': '5.4M', 'desc': 'EfficientNet-Lite1'},
    'efficientnet_lite2': {'params': '6.1M', 'desc': 'EfficientNet-Lite2'},

    # MNASNet 系列
    'mnasnet_050': {'params': '1.0M', 'desc': 'MNASNet 0.5x'},
    'mnasnet_100': {'params': '4.4M', 'desc': 'MNASNet 1.0x'},

    # ResNet 系列 (标准卷积，用于验证代码正确性)
    'resnet18': {'params': '11.7M', 'desc': 'ResNet-18 (standard conv, for debugging)'},
    'resnet34': {'params': '21.8M', 'desc': 'ResNet-34 (standard conv, for debugging)'},
    'resnet50': {'params': '25.6M', 'desc': 'ResNet-50 (standard conv, for debugging)'},
}

# ============== 默认 Anchor 配置 ==============
DEFAULT_ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],       # Small objects (stride 8)
    [[30, 61], [62, 45], [59, 119]],      # Medium objects (stride 16)
    [[116, 90], [156, 198], [373, 326]]   # Large objects (stride 32)
]


def list_available_backbones():
    """打印所有可用的 backbone"""
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
    """验证 backbone 名称是否有效"""
    if name in AVAILABLE_BACKBONES:
        return True
    # 尝试从 timm 加载
    try:
        model = timm.create_model(name, pretrained=False, features_only=True, out_indices=(2, 3, 4))
        del model
        return True
    except Exception:
        return False


def build_config(args) -> dict:
    """根据命令行参数构建配置字典"""
    config = {
        'model': {
            'backbone': args.backbone,
            'pretrained': not args.no_pretrained,
            'num_classes': 80,  # 会被数据集覆盖
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train MobileNet-YOLO Object Detection')

    # ===== Backbone 选择 =====
    parser.add_argument('--list-backbones', action='store_true',
                        help='List all available backbones and exit')
    parser.add_argument('--backbone', type=str, default='mobilenetv3_large_100',
                        help='Backbone model name (default: mobilenetv3_large_100)')

    # ===== 数据相关 =====
    parser.add_argument('--data', type=str, required=False, default='',
                        help='Dataset YAML file path (required for training)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers (default: 8)')

    # ===== 模型相关 =====
    parser.add_argument('--fpn-channels', type=int, default=128,
                        help='FPN output channels (default: 128, suitable for lightweight backbones)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained backbone weights')

    # ===== 训练相关 =====
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help='Weight decay (default: 0.00005, lower for lightweight models)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping max norm (default: 10.0)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='LR scheduler: cosine or plateau (default: cosine)')

    # ===== 损失权重 =====
    parser.add_argument('--lambda-box', type=float, default=7.5,
                        help='Box loss weight (default: 7.5, higher for CIoU loss)')
    parser.add_argument('--lambda-obj', type=float, default=1.0,
                        help='Objectness loss weight (default: 1.0)')
    parser.add_argument('--lambda-cls', type=float, default=0.5,
                        help='Classification loss weight (default: 0.5)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor for classification (default: 0.0, typical: 0.05-0.1)')

    # ===== 高级特性 =====
    parser.add_argument('--ema', action='store_true',
                        help='Use Exponential Moving Average')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau (epochs without improvement)')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='Factor to reduce learning rate (default: 0.1)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping after N LR reductions (0=disabled, default: 0)')

    # ===== 评估相关 =====
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluate every N epochs (default: 1)')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='Confidence threshold for evaluation (default: 0.001)')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                        help='IoU threshold for NMS (default: 0.6)')
    parser.add_argument('--plot-samples', type=int, default=16,
                        help='Number of detection samples to plot (default: 16)')

    # ===== 输出相关 =====
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Save directory (default: runs/train)')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name (default: exp)')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from experiment name (e.g., exp, exp_1)')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (default: auto)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')

    # ===== Anchor 相关 =====
    parser.add_argument('--auto-anchor', action='store_true', default=True,
                        help='Auto compute anchors from dataset (default: True)')
    parser.add_argument('--no-auto-anchor', action='store_true',
                        help='Disable auto anchor computation, use default anchors')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    scaler=None, ema=None, accumulator=None, grad_clip=10.0):
    """训练一个 epoch"""
    model.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    total_loss = 0
    loss_components = {}

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        # 移动 targets 到 device
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        # 前向传播 (AMP)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            predictions, anchors = model(images, targets)
            loss_dict, loss = criterion(predictions, anchors, targets)

        # 反向传播
        if accumulator:
            loss = loss / accumulator.accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 优化器步进
        if not accumulator or accumulator.should_step(batch_idx):
            if scaler:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad()

            if ema:
                ema.update(model)

        # 记录日志
        total_loss += loss_dict['total_loss']
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'box': f"{loss_dict['box_loss']:.4f}",
            'obj': f"{loss_dict['obj_loss']:.4f}",
            'cls': f"{loss_dict['cls_loss']:.4f}"
        })

    # 平均 Loss
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, criterion, device, class_names,
             compute_metrics=True, save_dir=None, plot_samples=16):
    """验证模型"""
    model.eval()

    total_loss = 0
    loss_components = {}
    sample_images = []
    sample_preds = []
    sample_targets = []

    detection_metrics = DetectionMetrics(nc=len(class_names))

    pbar = tqdm(dataloader, desc='Validation')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        # 计算 Loss
        predictions_raw, anchors = model(images, targets)
        loss_dict, _ = criterion(predictions_raw, anchors, targets)

        total_loss += loss_dict['total_loss']
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

        if compute_metrics:
            # 推理预测 (包含 NMS)
            batch_dets = model.predict(images, conf_thresh=0.001, nms_thresh=0.6)

            # 格式化 targets
            batch_targets_formatted = []
            for t in targets:
                boxes = t['boxes'].cpu().numpy()
                labels = t['labels'].cpu().numpy()
                if len(boxes) > 0:
                    formatted = np.column_stack((labels, boxes))
                    batch_targets_formatted.append(formatted)
                else:
                    batch_targets_formatted.append(np.zeros((0, 5)))

            # 格式化 predictions
            batch_preds_formatted = []
            for det in batch_dets:
                img_preds = []
                if len(det['boxes']) > 0:
                    boxes = det['boxes'].cpu().numpy()
                    scores = det['scores'].cpu().numpy()
                    labels = det['labels'].cpu().numpy()
                    for b, s, l in zip(boxes, scores, labels):
                        img_preds.append({
                            'bbox': b,
                            'confidence': s,
                            'class_id': int(l)
                        })
                batch_preds_formatted.append(img_preds)

            detection_metrics.update(batch_preds_formatted, batch_targets_formatted)

            # 收集绘图样本
            if batch_idx == 0 and save_dir:
                n = min(plot_samples, len(images))
                for i in range(n):
                    sample_images.append(images[i].cpu().numpy())
                    sample_preds.append(batch_preds_formatted[i])
                    sample_targets.append(batch_targets_formatted[i])

    # 平均 Loss
    num_batches = len(dataloader)
    metrics = {'val_loss': total_loss / num_batches}
    metrics.update({k: v / num_batches for k, v in loss_components.items()})

    # 计算 mAP
    if compute_metrics:
        det_metrics = detection_metrics.compute_metrics()
        metrics.update(det_metrics)

        if save_dir and sample_images:
            plot_detection_samples(
                sample_images, sample_preds, sample_targets,
                class_names, save_dir, max_images=plot_samples
            )

    return metrics


def main():
    args = parse_args()

    # 处理 --list-backbones 参数
    if args.list_backbones:
        list_available_backbones()
        return

    # 检查必需参数
    if not args.data:
        print(colorstr('bright_red', "Error: --data is required"))
        print("Usage: python train.py --data path/to/dataset.yaml")
        return

    # 验证 backbone
    if not validate_backbone(args.backbone):
        print(colorstr('bright_red', f"Error: Unknown backbone '{args.backbone}'"))
        print("Use --list-backbones to see available options")
        return

    init_seeds(args.seed)

    # 目录设置
    if args.resume:
        # 恢复训练：使用指定的实验目录
        save_dir = Path(args.project) / args.resume
        if not save_dir.exists():
            print(colorstr('bright_red', f'Experiment not found: {save_dir}'))
            return
    else:
        # 新训练：创建新目录
        save_dir = Path(increment_path(Path(args.project) / args.name))
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)

    # 构建配置
    config = build_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')

    # 打印配置信息
    print(colorstr('bright_green', f'\nStarting training on {device}'))
    print(colorstr('bright_cyan', f'Backbone: {args.backbone}'))
    print(colorstr('bright_cyan', f'Image size: {args.img_size}x{args.img_size}'))
    print(colorstr('bright_cyan', f'FPN channels: {args.fpn_channels}'))
    if args.label_smoothing > 0:
        print(colorstr('bright_cyan', f'Label smoothing: {args.label_smoothing}'))

    # 加载数据集
    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    # 加载 YOLO 格式数据集
    data_yaml = args.data
    print(f"\nLoading dataset from: {data_yaml}")

    train_dataset = YOLODataset(data_yaml, 'train', train_transforms, args.img_size)
    val_dataset = YOLODataset(data_yaml, 'val', val_transforms, args.img_size)

    config['model']['num_classes'] = train_dataset.num_classes
    class_names = train_dataset.class_names

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Number of classes: {config['model']['num_classes']}")

    # 计算或加载 anchors
    if args.auto_anchor and not args.no_auto_anchor:
        print(colorstr('bright_cyan', '\nAuto-computing anchors...'))
        anchors = get_or_compute_anchors(
            save_dir=str(save_dir),
            yaml_path=data_yaml,
            img_size=args.img_size,
            n_anchors=9
        )
        config['anchors'] = anchors
        print(colorstr('bright_green', f'Using computed anchors: {anchors}'))
    else:
        print(colorstr('bright_yellow', f'Using default anchors: {config["anchors"]}'))

    # 创建模型
    model = MobileNetDetector(config).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 损失函数
    criterion = YOLOLoss(
        config['model']['num_classes'],
        model.anchor_generator,
        model.strides,
        config['model']['input_size'],
        config['train']['loss_weights'],
        label_smoothing=args.label_smoothing
    )

    # 训练工具
    plotter = TrainingPlotter(save_dir)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    ema = ModelEMA(model) if args.ema else None
    accumulator = GradientAccumulator(args.accumulation_steps) if args.accumulation_steps > 1 else None
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode='min',
        check_lr_reductions=True,
        max_lr_reductions=args.early_stopping,
        verbose=True
    ) if args.early_stopping > 0 else None

    # LR Scheduler setup
    # Warmup: 前 warmup_epochs 个 epoch 线性增加到目标学习率
    total_epochs = args.epochs
    warmup_epochs = args.warmup_epochs

    if args.scheduler == 'cosine':
        # Cosine Annealing with Warmup (recommended for lightweight models)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_type = 'cosine'
        print(colorstr('bright_cyan', f'Using Cosine Annealing scheduler with {warmup_epochs} warmup epochs'))

    else:  # plateau
        # ReduceLROnPlateau with Warmup (original behavior)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_type = 'plateau'
        print(colorstr('bright_cyan', f'Using ReduceLROnPlateau scheduler with {warmup_epochs} warmup epochs'))

    # ReduceLROnPlateau - only used when scheduler='plateau'
    lr_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.patience,
        min_lr=args.min_lr,
        verbose=True
    ) if args.scheduler == 'plateau' else None

    # 训练循环
    best_loss = float('inf')
    best_mAP = 0.0  # 最佳 mAP
    ema_val_loss = None  # EMA 平滑验证损失（用于稳定模型选择）
    ema_alpha = 0.3  # EMA 衰减系数，越小越平滑
    start_epoch = 0

    # 恢复训练
    if args.resume:
        ckpt_path = weights_dir / 'last.pt'
        if ckpt_path.exists():
            print(colorstr('bright_cyan', f'\nResuming from {ckpt_path}'))
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            if 'lr_plateau' in ckpt and lr_plateau is not None:
                lr_plateau.load_state_dict(ckpt['lr_plateau'])
            if 'scaler' in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt['scaler'])
            if 'ema' in ckpt and ema is not None:
                ema.ema.load_state_dict(ckpt['ema'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            best_mAP = ckpt.get('best_mAP', 0.0)
            ema_val_loss = ckpt.get('ema_val_loss', None)
            print(colorstr('bright_green', f'Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}, best_mAP={best_mAP:.4f}'))
        else:
            print(colorstr('bright_red', f'Checkpoint not found: {ckpt_path}'))
            return

    print(f"\nLogging to {save_dir}\n")

    for epoch in range(start_epoch, args.epochs):
        # 打印 epoch 信息和当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if epoch < args.warmup_epochs:
            epoch_color = 'bright_yellow'  # Warmup 阶段用黄色
        else:
            epoch_color = 'bright_cyan'    # 正常训练用青色
        print(f'\n{colorstr(epoch_color, "bold", f"Epoch {epoch}/{args.epochs-1}")} | LR: {current_lr:.2e}')

        # 训练
        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, ema=ema, accumulator=accumulator, grad_clip=args.grad_clip
        )

        # 验证
        compute_metrics = (epoch % args.eval_interval == 0) or (epoch == args.epochs - 1)
        val_model = ema.ema if ema else model

        val_metrics = validate(
            val_model, val_loader, criterion, device, class_names,
            compute_metrics=compute_metrics, save_dir=save_dir, plot_samples=args.plot_samples
        )

        # 更新学习率调度
        current_val_loss = val_metrics['val_loss']
        lr_reduced = False

        if scheduler_type == 'cosine':
            # Cosine scheduler updates per epoch
            scheduler.step()
        else:  # plateau
            if epoch < args.warmup_epochs:
                # Warmup 阶段
                scheduler.step()
            else:
                # Warmup 结束后，由 ReduceLROnPlateau 根据验证损失调整
                lr_reduced = lr_plateau.step(current_val_loss)

        # 记录指标
        metrics_all = {'train_loss': train_loss, **val_metrics}
        plotter.update(epoch, metrics_all)
        plotter.save_metrics_csv()

        # 计算 EMA 平滑验证损失（current_val_loss 已在上面获取）
        if ema_val_loss is None:
            ema_val_loss = current_val_loss
        else:
            ema_val_loss = ema_alpha * current_val_loss + (1 - ema_alpha) * ema_val_loss

        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {current_val_loss:.4f} (EMA: {ema_val_loss:.4f})")
        if compute_metrics and 'mAP@0.5' in val_metrics:
            print(f"mAP@0.5: {val_metrics['mAP@0.5']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")

        # 保存最佳模型（混合策略）
        # - 当 mAP 评估启用时：mAP 提升或 val_loss 下降都会更新模型
        # - 当 mAP 评估未启用时：仅根据 val_loss 更新
        save_best = False
        save_reason = ""
        current_mAP = val_metrics.get('mAP@0.5', None)

        if compute_metrics and current_mAP is not None:
            # mAP 评估已启用：mAP 提升或 val_loss 下降都更新
            if current_mAP > best_mAP:
                save_best = True
                save_reason = f"mAP improved: {best_mAP:.4f} -> {current_mAP:.4f}"
                best_mAP = current_mAP
            if ema_val_loss < best_loss:
                save_best = True
                if save_reason:
                    save_reason += f", val_loss (EMA) improved: {best_loss:.4f} -> {ema_val_loss:.4f}"
                else:
                    save_reason = f"val_loss (EMA) improved: {best_loss:.4f} -> {ema_val_loss:.4f}"
                best_loss = ema_val_loss
        else:
            # mAP 评估未启用：仅依赖 val_loss
            if ema_val_loss < best_loss:
                save_best = True
                save_reason = f"val_loss (EMA) improved: {best_loss:.4f} -> {ema_val_loss:.4f}"
                best_loss = ema_val_loss

        if save_best:
            torch.save({
                'model': val_model.state_dict(),
                'best_loss': best_loss,
                'best_mAP': best_mAP,
                'epoch': epoch,
                'args': vars(args)
            }, weights_dir / 'best.pt')
            print(colorstr('bright_green', f"New best model saved: {save_reason}"))

        # 保存 Last（包含完整状态用于恢复训练）
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scheduler_type': scheduler_type,
            'epoch': epoch,
            'best_loss': best_loss,
            'best_mAP': best_mAP,
            'ema_val_loss': ema_val_loss,
            'args': vars(args)
        }
        if lr_plateau is not None:
            ckpt['lr_plateau'] = lr_plateau.state_dict()
        if scaler is not None:
            ckpt['scaler'] = scaler.state_dict()
        if ema is not None:
            ckpt['ema'] = ema.ema.state_dict()
        torch.save(ckpt, weights_dir / 'last.pt')

        # 早停检查（基于学习率降低次数，仅在 plateau scheduler 时有效）
        if early_stopping and lr_plateau is not None:
            should_stop = early_stopping.step(
                ema_val_loss,
                lr_reduced=lr_reduced,
                num_lr_reductions=lr_plateau.num_lr_reductions
            )
            if should_stop:
                print(colorstr('red', 'bold', '\nEarly stopping triggered!'))
                break

    plotter.plot_training_curves()
    print(f"\nTraining complete. Results saved to {save_dir}")
    print(f"Best val_loss (EMA): {best_loss:.4f}, Best mAP@0.5: {best_mAP:.4f}")


if __name__ == '__main__':
    main()
