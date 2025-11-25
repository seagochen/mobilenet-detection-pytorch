"""
Training script for MobileNet-YOLO (Enhanced Version)
Matches features of YOLOv2-PyTorch training script
"""
import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mobilenet_ssdlite.models import MobileNetYOLO
from mobilenet_ssdlite.models.loss import YOLOLoss
from mobilenet_ssdlite.utils import DetectionDataset, YOLODataset, collate_fn, get_transforms

# 引入从 YOLOv2 移植过来的工具类
from mobilenet_ssdlite.utils.metrics import DetectionMetrics
from mobilenet_ssdlite.utils.plots import TrainingPlotter, plot_detection_samples, plot_labels_distribution
from mobilenet_ssdlite.utils.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelEMA, GradientAccumulator, WarmupScheduler
)
from mobilenet_ssdlite.utils.general import init_seeds, colorstr, increment_path, check_img_size


def parse_args():
    parser = argparse.ArgumentParser(description='Train MobileNet-YOLO')

    # 基础配置
    parser.add_argument('--config', type=str, default='configs/yolo_format.yaml', help='Path to config file')
    parser.add_argument('--project', type=str, default='runs/train', help='Save directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='', help='Device (0 or cpu)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # 训练参数 (覆盖 yaml)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)

    # 高级特性
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--ema', action='store_true', help='Use Model EMA')
    parser.add_argument('--amp', action='store_true', help='Use Mixed Precision')
    parser.add_argument('--accumulation-steps', type=int, default=1)

    # 评估
    parser.add_argument('--eval-interval', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--plot-samples', type=int, default=16, help='Number of samples to plot')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, ema=None, accumulator=None, grad_clip=10.0):
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
        with torch.cuda.amp.autocast(enabled=scaler is not None):
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
            'cls': f"{loss_dict['cls_loss']:.4f}"
        })

    # 平均 Loss
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, criterion, device, class_names, compute_metrics=True, save_dir=None, plot_samples=16):
    model.eval()

    total_loss = 0
    loss_components = {}

    # 收集用于计算 mAP 的数据
    all_predictions = []  # List of dicts
    all_targets = []      # List of [class, x, y, x, y]

    # 收集用于绘图的样本
    sample_images = []
    sample_preds = []
    sample_targets = []

    detection_metrics = DetectionMetrics(nc=len(class_names))

    pbar = tqdm(dataloader, desc='Validation')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        # 移动 targets
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

            # 处理 targets 格式以适配 DetectionMetrics
            # DetectionMetrics 期望 targets 为 list of [N, 5] numpy array (cls, x1, y1, x2, y2)
            batch_targets_formatted = []
            for t in targets:
                boxes = t['boxes'].cpu().numpy()
                labels = t['labels'].cpu().numpy()
                if len(boxes) > 0:
                    # 合并 class 和 boxes -> [cls, x1, y1, x2, y2]
                    formatted = np.column_stack((labels, boxes))
                    batch_targets_formatted.append(formatted)
                else:
                    batch_targets_formatted.append(np.zeros((0, 5)))

            # 处理 predictions 格式
            # DetectionMetrics 期望 predictions 为 list of dicts with 'bbox', 'confidence', 'class_id'
            batch_preds_formatted = []
            for det in batch_dets:
                # det is {'boxes': tensor, 'scores': tensor, 'labels': tensor}
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

            # 更新指标
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

        # 绘图
        if save_dir and sample_images:
            plot_detection_samples(
                sample_images, sample_preds, sample_targets,
                class_names, save_dir, max_images=plot_samples
            )

    return metrics


def main():
    args = parse_args()
    init_seeds(args.seed)

    # 目录设置
    save_dir = Path(increment_path(Path(args.project) / args.name))
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 覆盖 Config 参数
    config['train']['epochs'] = args.epochs
    config['train']['batch_size'] = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(colorstr('bright_green', f'Starting training on {device}'))

    # 加载数据集
    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    # 自动检测 YOLO 格式
    if 'yaml' in config['data']:
        data_yaml = config['data']['yaml']
        print(f"Using YOLO dataset config: {data_yaml}")
        train_dataset = YOLODataset(data_yaml, 'train', train_transforms, config['model']['input_size'][0])
        val_dataset = YOLODataset(data_yaml, 'val', val_transforms, config['model']['input_size'][0])
        config['model']['num_classes'] = train_dataset.num_classes
        class_names = train_dataset.class_names
    else:
        # Legacy format
        train_dataset = DetectionDataset(config['data']['train'], None, train_transforms, config['model']['input_size'][0])
        val_dataset = DetectionDataset(config['data']['val'], None, val_transforms, config['model']['input_size'][0])
        class_names = config['data'].get('names', [f'class_{i}' for i in range(config['model']['num_classes'])])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Number of classes: {config['model']['num_classes']}")

    # 模型
    model = MobileNetYOLO(config).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['optimizer']['lr'],
                            weight_decay=config['train']['optimizer']['weight_decay'])

    # 损失函数
    criterion = YOLOLoss(config['model']['num_classes'], model.anchor_generator, model.strides,
                         config['model']['input_size'], config['train']['loss_weights'])

    # 工具
    plotter = TrainingPlotter(save_dir)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    ema = ModelEMA(model) if args.ema else None
    accumulator = GradientAccumulator(args.accumulation_steps) if args.accumulation_steps > 1 else None
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # LR Scheduler
    # 简单的 Warmup + Cosine
    lf = lambda x: ((1 - np.cos(x * np.pi / args.epochs)) / 2) * (1 - 0.1) + 0.1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 训练循环
    best_map = 0.0

    print(f"Logging to {save_dir}")

    for epoch in range(args.epochs):
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

        scheduler.step()

        # 记录合并指标
        metrics_all = {'train_loss': train_loss, **val_metrics}
        plotter.update(epoch, metrics_all)
        plotter.save_metrics_csv()  # 实时保存 CSV

        # 打印
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        if compute_metrics and 'mAP@0.5' in val_metrics:
            print(f"Metrics: mAP@0.5: {val_metrics['mAP@0.5']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")

            # 保存最佳模型 (根据 mAP)
            if val_metrics['mAP@0.5'] > best_map:
                best_map = val_metrics['mAP@0.5']
                torch.save({'model': val_model.state_dict(), 'best_map': best_map, 'epoch': epoch}, weights_dir / 'best.pt')
                print(colorstr('bright_green', f"New best model: {best_map:.4f}"))

        # 保存 Last
        torch.save({'model': model.state_dict(), 'epoch': epoch}, weights_dir / 'last.pt')

        # 早停
        if early_stopping.step(val_metrics['val_loss']):
            print("Early stopping triggered")
            break

    plotter.plot_training_curves()
    print(f"Training complete. Results saved to {save_dir}")


if __name__ == '__main__':
    main()
