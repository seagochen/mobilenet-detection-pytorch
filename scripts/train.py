"""
Training script for MobileNet-YOLO
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mobilenet_ssdlite.models import MobileNetYOLO
from mobilenet_ssdlite.models.loss import YOLOLoss
from mobilenet_ssdlite.utils import DetectionDataset, YOLODataset, collate_fn, get_transforms
from mobilenet_ssdlite.utils.visualize import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='Train MobileNet-YOLO')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_box_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        # Move targets to device
        for i in range(len(targets)):
            targets[i]['boxes'] = targets[i]['boxes'].to(device)
            targets[i]['labels'] = targets[i]['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions, anchors = model(images, targets)

        # Compute loss
        loss_dict, loss = criterion(predictions, anchors, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss_dict['total_loss']
        total_box_loss += loss_dict['box_loss']
        total_obj_loss += loss_dict['obj_loss']
        total_cls_loss += loss_dict['cls_loss']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'box': f"{loss_dict['box_loss']:.4f}",
            'obj': f"{loss_dict['obj_loss']:.4f}",
            'cls': f"{loss_dict['cls_loss']:.4f}"
        })

    # Average metrics
    num_batches = len(dataloader)
    avg_metrics = {
        'total_loss': total_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches
    }

    return avg_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()

    total_loss = 0
    total_box_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0

    pbar = tqdm(dataloader, desc='Validation')

    for images, targets in pbar:
        images = images.to(device)

        # Move targets to device
        for i in range(len(targets)):
            targets[i]['boxes'] = targets[i]['boxes'].to(device)
            targets[i]['labels'] = targets[i]['labels'].to(device)

        # Forward pass
        predictions, anchors = model(images, targets)

        # Compute loss
        loss_dict, _ = criterion(predictions, anchors, targets)

        # Update metrics
        total_loss += loss_dict['total_loss']
        total_box_loss += loss_dict['box_loss']
        total_obj_loss += loss_dict['obj_loss']
        total_cls_loss += loss_dict['cls_loss']

        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}"
        })

    # Average metrics
    num_batches = len(dataloader)
    avg_metrics = {
        'total_loss': total_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches
    }

    return avg_metrics


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = MobileNetYOLO(config)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    print("Loading datasets...")
    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    # Check if using YOLO format (yaml file) or legacy format
    data_config = config['data']
    if 'yaml' in data_config:
        # YOLO format with yaml config
        yaml_path = data_config['yaml']
        print(f"Using YOLO format dataset: {yaml_path}")

        train_dataset = YOLODataset(
            yaml_path=yaml_path,
            split='train',
            transforms=train_transforms,
            img_size=config['model']['input_size'][0]
        )

        val_dataset = YOLODataset(
            yaml_path=yaml_path,
            split='val',
            transforms=val_transforms,
            img_size=config['model']['input_size'][0]
        )

        # Update num_classes from dataset if not specified
        if config['model']['num_classes'] != train_dataset.num_classes:
            print(f"Updating num_classes from {config['model']['num_classes']} to {train_dataset.num_classes}")
            config['model']['num_classes'] = train_dataset.num_classes

    else:
        # Legacy format with separate directories
        print("Using legacy format dataset")
        train_dataset = DetectionDataset(
            data_dir=config['data']['train'],
            annotation_file=None,
            transforms=train_transforms,
            img_size=config['model']['input_size'][0]
        )

        val_dataset = DetectionDataset(
            data_dir=config['data']['val'],
            annotation_file=None,
            transforms=val_transforms,
            img_size=config['model']['input_size'][0]
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create loss function
    criterion = YOLOLoss(
        num_classes=config['model']['num_classes'],
        anchors=model.anchor_generator,
        strides=model.strides,
        input_size=config['model']['input_size'],
        loss_weights=config['train']['loss_weights']
    )

    # Create optimizer
    optimizer_config = config['train']['optimizer']
    if optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=0.9,
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")

    # Create scheduler
    scheduler_config = config['train']['scheduler']
    if scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
    else:
        scheduler = None

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Create checkpoint directory
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create tensorboard writer
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training history
    history = {
        'total_loss': [],
        'box_loss': [],
        'obj_loss': [],
        'cls_loss': [],
        'val_total_loss': []
    }

    # Training loop
    num_epochs = config['train']['epochs']
    best_val_loss = float('inf')

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )

        # Log training metrics
        print(f"\nEpoch {epoch} - Train Loss: {train_metrics['total_loss']:.4f}")
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
            history[key].append(value)

        # Validate
        if len(val_dataset) > 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch} - Val Loss: {val_metrics['total_loss']:.4f}")

            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)

            history['val_total_loss'].append(val_metrics['total_loss'])

            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': best_val_loss,
                }, os.path.join(checkpoint_dir, 'best.pth'))
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        # Update scheduler
        if scheduler:
            scheduler.step()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, os.path.join(checkpoint_dir, 'final.pth'))

    # Plot training curves
    plot_training_curves(history, os.path.join(log_dir, 'training_curves.png'))

    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
