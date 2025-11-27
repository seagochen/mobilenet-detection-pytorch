"""
Training pipeline for MobileNet-YOLO object detection.
Encapsulates the training loop and related utilities.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import math
from typing import Dict, Any, Optional, Tuple, List

from ..models import MobileNetDetector
from ..models.loss import DetectionLoss, AutomaticWeightedLoss
from ..utils import (
    YOLODataset, collate_fn, get_transforms,
    ReduceLROnPlateau, EarlyStopping, ModelEMA, GradientAccumulator,
    TrainingPlotter, colorstr, get_or_compute_anchors
)

from .config import DEFAULT_ANCHORS
from .evaluator import validate


class Trainer:
    """
    Training pipeline for object detection models.

    Handles the complete training workflow including:
    - Model and optimizer setup
    - Learning rate scheduling with warmup
    - Mixed precision training (AMP)
    - Exponential moving average (EMA)
    - Gradient accumulation
    - Checkpointing and resuming
    - Early stopping
    """

    def __init__(
        self,
        config: Dict[str, Any],
        args,
        save_dir: Path,
        device: torch.device
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
            args: Command line arguments
            save_dir: Directory to save checkpoints and logs
            device: Training device (cuda/cpu)
        """
        self.config = config
        self.args = args
        self.save_dir = save_dir
        self.device = device
        self.weights_dir = save_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        # Will be initialized in setup()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.auto_loss_wrapper = None  # Automatic loss weighting
        self.scheduler = None
        self.lr_plateau = None
        self.scaler = None
        self.ema = None
        self.accumulator = None
        self.early_stopping = None
        self.plotter = None

        self.train_loader = None
        self.val_loader = None
        self.class_names = None

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_mAP = 0.0
        self.ema_val_loss = None
        self.ema_alpha = 0.3
        self.scheduler_type = 'cosine'

    def setup_data(self) -> None:
        """Setup datasets and dataloaders."""
        args = self.args
        config = self.config

        train_transforms = get_transforms(config, is_train=True)
        val_transforms = get_transforms(config, is_train=False)

        print(f"\nLoading dataset from: {args.data}")

        train_dataset = YOLODataset(args.data, 'train', train_transforms, args.img_size)
        val_dataset = YOLODataset(args.data, 'val', val_transforms, args.img_size)

        config['model']['num_classes'] = train_dataset.num_classes
        self.class_names = train_dataset.class_names

        self.train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
            drop_last=True  # Drop incomplete last batch to stabilize BatchNorm
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
        )

        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Val dataset: {len(val_dataset)} images")
        print(f"Number of classes: {config['model']['num_classes']}")

    def setup_anchors(self) -> None:
        """Compute or load anchors."""
        args = self.args
        config = self.config

        if args.auto_anchor and not args.no_auto_anchor:
            print(colorstr('bright_cyan', '\nAuto-computing anchors...'))
            anchors = get_or_compute_anchors(
                save_dir=str(self.save_dir),
                yaml_path=args.data,
                img_size=args.img_size,
                n_anchors=9
            )
            config['anchors'] = anchors
            print(colorstr('bright_green', f'Using computed anchors: {anchors}'))
        else:
            print(colorstr('bright_yellow', f'Using default anchors: {config["anchors"]}'))

    def setup_model(self) -> None:
        """Create model, optimizer, loss function and training utilities."""
        args = self.args
        config = self.config

        # Create model
        self.model = MobileNetDetector(config).to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Loss function
        self.criterion = DetectionLoss(
            config['model']['num_classes'],
            self.model.anchor_generator,
            self.model.strides,
            config['model']['input_size'],
            config['train']['loss_weights'],
            label_smoothing=args.label_smoothing
        )

        # Automatic loss weighting (if enabled)
        params_to_optimize = list(self.model.parameters())
        if getattr(args, 'auto_loss', False):
            self.auto_loss_wrapper = AutomaticWeightedLoss(num_losses=3).to(self.device)
            params_to_optimize += list(self.auto_loss_wrapper.parameters())
            print(colorstr('bright_cyan', 'Using Automatic Loss Weighting (Uncertainty-based)'))

        # Optimizer (includes auto_loss params if enabled)
        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Training utilities
        self.plotter = TrainingPlotter(self.save_dir)
        self.scaler = torch.amp.GradScaler('cuda') if args.amp else None
        self.ema = ModelEMA(self.model) if args.ema else None
        self.accumulator = GradientAccumulator(args.accumulation_steps) if args.accumulation_steps > 1 else None

        if args.early_stopping > 0:
            self.early_stopping = EarlyStopping(
                patience=args.patience,
                mode='min',
                check_lr_reductions=True,
                max_lr_reductions=args.early_stopping,
                verbose=True
            )

        self._setup_scheduler()

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        args = self.args
        total_epochs = args.epochs
        warmup_epochs = args.warmup_epochs

        if args.scheduler == 'cosine':
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            self.scheduler_type = 'cosine'
            print(colorstr('bright_cyan', f'Using Cosine Annealing scheduler with {warmup_epochs} warmup epochs'))

        else:  # plateau
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            self.scheduler_type = 'plateau'
            self.lr_plateau = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=args.lr_factor,
                patience=args.patience,
                min_lr=args.min_lr,
                verbose=True
            )
            print(colorstr('bright_cyan', f'Using ReduceLROnPlateau scheduler with {warmup_epochs} warmup epochs'))

    def resume(self) -> bool:
        """
        Resume training from checkpoint.

        Returns:
            True if successfully resumed, False otherwise
        """
        ckpt_path = self.weights_dir / 'last.pt'
        if not ckpt_path.exists():
            print(colorstr('bright_red', f'Checkpoint not found: {ckpt_path}'))
            return False

        print(colorstr('bright_cyan', f'\nResuming from {ckpt_path}'))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'lr_plateau' in ckpt and self.lr_plateau is not None:
            self.lr_plateau.load_state_dict(ckpt['lr_plateau'])
        if 'scaler' in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        if 'ema' in ckpt and self.ema is not None:
            self.ema.ema.load_state_dict(ckpt['ema'])
        if 'auto_loss_wrapper' in ckpt and self.auto_loss_wrapper is not None:
            self.auto_loss_wrapper.load_state_dict(ckpt['auto_loss_wrapper'])

        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.best_mAP = ckpt.get('best_mAP', 0.0)
        self.ema_val_loss = ckpt.get('ema_val_loss', None)

        print(colorstr('bright_green',
            f'Resumed from epoch {self.start_epoch}, best_loss={self.best_loss:.4f}, best_mAP={self.best_mAP:.4f}'))
        return True

    def train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, loss_components_dict)
        """
        self.model.train()
        args = self.args

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        total_loss = 0
        loss_components = {}

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)

            for t in targets:
                t['boxes'] = t['boxes'].to(self.device)
                t['labels'] = t['labels'].to(self.device)

            # Forward pass (AMP)
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                predictions, anchors = self.model(images, targets)
                loss_dict, loss, loss_tensors = self.criterion(predictions, anchors, targets)

                # Apply automatic loss weighting if enabled
                if self.auto_loss_wrapper is not None:
                    # Use tensor losses for proper gradient computation
                    loss, weight_dict = self.auto_loss_wrapper(
                        loss_tensors['box_loss'],
                        loss_tensors['obj_loss'],
                        loss_tensors['cls_loss']
                    )
                    loss_dict['total_loss'] = loss.item()
                    # Store weights for logging
                    loss_dict.update(weight_dict)

            # Backward pass
            if self.accumulator:
                loss = loss / self.accumulator.accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if not self.accumulator or self.accumulator.should_step(batch_idx):
                if self.scaler:
                    if args.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.ema:
                    self.ema.update(self.model)

            # Logging
            total_loss += loss_dict['total_loss']
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v

            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'box': f"{loss_dict['box_loss']:.4f}",
                'obj': f"{loss_dict['obj_loss']:.4f}",
                'cls': f"{loss_dict['cls_loss']:.4f}"
            })

        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return avg_loss, avg_components

    def save_checkpoint(self, epoch: int, is_best: bool = False, save_reason: str = "") -> None:
        """Save model checkpoint."""
        val_model = self.ema.ema if self.ema else self.model

        if is_best:
            torch.save({
                'model': val_model.state_dict(),
                'best_loss': self.best_loss,
                'best_mAP': self.best_mAP,
                'epoch': epoch,
                'args': vars(self.args)
            }, self.weights_dir / 'best.pt')
            print(colorstr('bright_green', f"New best model saved: {save_reason}"))

        # Save last checkpoint (full state for resuming)
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scheduler_type': self.scheduler_type,
            'epoch': epoch,
            'best_loss': self.best_loss,
            'best_mAP': self.best_mAP,
            'ema_val_loss': self.ema_val_loss,
            'args': vars(self.args)
        }
        if self.lr_plateau is not None:
            ckpt['lr_plateau'] = self.lr_plateau.state_dict()
        if self.scaler is not None:
            ckpt['scaler'] = self.scaler.state_dict()
        if self.ema is not None:
            ckpt['ema'] = self.ema.ema.state_dict()
        if self.auto_loss_wrapper is not None:
            ckpt['auto_loss_wrapper'] = self.auto_loss_wrapper.state_dict()
        torch.save(ckpt, self.weights_dir / 'last.pt')

    def train(self) -> None:
        """Run the complete training loop."""
        args = self.args

        print(f"\nLogging to {self.save_dir}\n")

        for epoch in range(self.start_epoch, args.epochs):
            # Print epoch info
            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch < args.warmup_epochs:
                epoch_color = 'bright_yellow'
            else:
                epoch_color = 'bright_cyan'
            print(f'\n{colorstr(epoch_color, "bold", f"Epoch {epoch}/{args.epochs-1}")} | LR: {current_lr:.2e}')

            # Train
            train_loss, train_components = self.train_one_epoch(epoch)

            # Validate
            compute_metrics = (epoch % args.eval_interval == 0) or (epoch == args.epochs - 1)
            val_model = self.ema.ema if self.ema else self.model

            val_metrics = validate(
                val_model, self.val_loader, self.criterion, self.device, self.class_names,
                compute_metrics=compute_metrics, save_dir=self.save_dir, plot_samples=args.plot_samples
            )

            # Update learning rate
            current_val_loss = val_metrics['val_loss']
            lr_reduced = False

            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            else:  # plateau
                if epoch < args.warmup_epochs:
                    self.scheduler.step()
                else:
                    lr_reduced = self.lr_plateau.step(current_val_loss)

            # Record metrics
            metrics_all = {'train_loss': train_loss, **val_metrics}
            self.plotter.update(epoch, metrics_all)
            self.plotter.save_metrics_csv()

            # Update EMA validation loss
            if self.ema_val_loss is None:
                self.ema_val_loss = current_val_loss
            else:
                self.ema_val_loss = self.ema_alpha * current_val_loss + (1 - self.ema_alpha) * self.ema_val_loss

            # Print results
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {current_val_loss:.4f} (EMA: {self.ema_val_loss:.4f})")
            if compute_metrics and 'mAP@0.5' in val_metrics:
                print(f"mAP@0.5: {val_metrics['mAP@0.5']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")

            # Print auto loss weights if enabled
            if self.auto_loss_wrapper is not None:
                weights = self.auto_loss_wrapper.get_weights()
                print(f"Auto weights: box={weights['w_box']:.3f}, obj={weights['w_obj']:.3f}, cls={weights['w_cls']:.3f}")

            # Check for best model
            save_best, save_reason = self._check_best_model(val_metrics, compute_metrics)

            # Save checkpoints
            self.save_checkpoint(epoch, is_best=save_best, save_reason=save_reason)

            # Early stopping check
            if self.early_stopping and self.lr_plateau is not None:
                should_stop = self.early_stopping.step(
                    self.ema_val_loss,
                    lr_reduced=lr_reduced,
                    num_lr_reductions=self.lr_plateau.num_lr_reductions
                )
                if should_stop:
                    print(colorstr('red', 'bold', '\nEarly stopping triggered!'))
                    break

        self.plotter.plot_training_curves()
        print(f"\nTraining complete. Results saved to {self.save_dir}")
        print(f"Best val_loss (EMA): {self.best_loss:.4f}, Best mAP@0.5: {self.best_mAP:.4f}")

    def _check_best_model(self, val_metrics: Dict[str, float], compute_metrics: bool) -> Tuple[bool, str]:
        """Check if current model is the best and update tracking."""
        save_best = False
        save_reason = ""
        current_mAP = val_metrics.get('mAP@0.5', None)

        if compute_metrics and current_mAP is not None:
            if current_mAP > self.best_mAP:
                save_best = True
                save_reason = f"mAP improved: {self.best_mAP:.4f} -> {current_mAP:.4f}"
                self.best_mAP = current_mAP
            if self.ema_val_loss < self.best_loss:
                save_best = True
                if save_reason:
                    save_reason += f", val_loss (EMA) improved: {self.best_loss:.4f} -> {self.ema_val_loss:.4f}"
                else:
                    save_reason = f"val_loss (EMA) improved: {self.best_loss:.4f} -> {self.ema_val_loss:.4f}"
                self.best_loss = self.ema_val_loss
        else:
            if self.ema_val_loss < self.best_loss:
                save_best = True
                save_reason = f"val_loss (EMA) improved: {self.best_loss:.4f} -> {self.ema_val_loss:.4f}"
                self.best_loss = self.ema_val_loss

        return save_best, save_reason
