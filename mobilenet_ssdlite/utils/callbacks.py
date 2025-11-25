"""
训练回调函数模块
包含：ReduceLROnPlateau, EarlyStopping, ModelEMA 等
"""

import torch
import torch.nn as nn
from copy import deepcopy
import math


class ReduceLROnPlateau:
    """
    当验证指标停止改善时降低学习率

    Args:
        optimizer: 优化器
        mode: 'min' 表示指标越小越好，'max' 表示指标越大越好
        factor: 学习率衰减因子，new_lr = lr * factor
        patience: 容忍多少个epoch指标不改善
        min_lr: 学习率下限
        threshold: 判断改善的阈值
        verbose: 是否打印信息
    """

    def __init__(
        self,
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-7,
        threshold=1e-4,
        verbose=True
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.verbose = verbose

        self.best = None
        self.num_bad_epochs = 0
        self.num_lr_reductions = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

        self._init_is_better()

    def _init_is_better(self):
        if self.mode == 'min':
            self.is_better = lambda a, best: a < best - self.threshold
        else:
            self.is_better = lambda a, best: a > best + self.threshold

    def step(self, metrics):
        """
        根据指标更新学习率

        Args:
            metrics: 当前epoch的验证指标

        Returns:
            bool: 是否降低了学习率
        """
        current = float(metrics)
        reduced = False

        if self.best is None:
            self.best = current
            self.num_bad_epochs = 0
        elif self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            reduced = self._reduce_lr()
            self.num_bad_epochs = 0

        return reduced

    def _reduce_lr(self):
        """降低学习率"""
        reduced = False
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)

            if old_lr > self.min_lr:
                param_group['lr'] = new_lr
                reduced = True
                self._last_lr[i] = new_lr

                if self.verbose:
                    print(f'  ↓ Reducing learning rate: {old_lr:.2e} → {new_lr:.2e}')

        if reduced:
            self.num_lr_reductions += 1

        return reduced

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'num_lr_reductions': self.num_lr_reductions,
            '_last_lr': self._last_lr
        }

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_lr_reductions = state_dict['num_lr_reductions']
        self._last_lr = state_dict['_last_lr']


class EarlyStopping:
    """
    早停机制：当验证指标连续多次不改善时停止训练

    Args:
        patience: 容忍多少个epoch指标不改善（或多少次学习率降低后仍不改善）
        mode: 'min' 表示指标越小越好，'max' 表示指标越大越好
        min_delta: 最小改善阈值
        check_lr_reductions: 如果为True，则基于学习率降低次数判断是否停止
        max_lr_reductions: 最大学习率降低次数（配合 check_lr_reductions 使用）
        verbose: 是否打印信息
    """

    def __init__(
        self,
        patience=10,
        mode='min',
        min_delta=0.0,
        check_lr_reductions=False,
        max_lr_reductions=3,
        verbose=True
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.check_lr_reductions = check_lr_reductions
        self.max_lr_reductions = max_lr_reductions
        self.verbose = verbose

        self.best = None
        self.counter = 0
        self.should_stop = False

        self._init_is_better()

    def _init_is_better(self):
        if self.mode == 'min':
            self.is_better = lambda a, best: a < best - self.min_delta
        else:
            self.is_better = lambda a, best: a > best + self.min_delta

    def step(self, metrics, lr_reduced=False, num_lr_reductions=0):
        """
        检查是否应该停止训练

        Args:
            metrics: 当前epoch的验证指标
            lr_reduced: 本次是否降低了学习率（配合 check_lr_reductions）
            num_lr_reductions: 学习率降低总次数

        Returns:
            bool: 是否应该停止训练
        """
        current = float(metrics)

        # 如果使用学习率降低次数来判断
        if self.check_lr_reductions:
            if lr_reduced:
                # 检查降低学习率后是否有改善
                if self.best is not None and not self.is_better(current, self.best):
                    if num_lr_reductions >= self.max_lr_reductions:
                        self.should_stop = True
                        if self.verbose:
                            print(f'  ✗ Early stopping: {self.max_lr_reductions} LR reductions without improvement')

            # 更新 best
            if self.best is None or self.is_better(current, self.best):
                self.best = current
        else:
            # 使用传统的 patience 计数
            if self.best is None:
                self.best = current
                self.counter = 0
            elif self.is_better(current, self.best):
                self.best = current
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose and self.counter > 0:
                    print(f'  ⚠ EarlyStopping counter: {self.counter}/{self.patience}')

                if self.counter >= self.patience:
                    self.should_stop = True
                    if self.verbose:
                        print(f'  ✗ Early stopping: No improvement for {self.patience} epochs')

        return self.should_stop

    def state_dict(self):
        return {
            'best': self.best,
            'counter': self.counter,
            'should_stop': self.should_stop
        }

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.counter = state_dict['counter']
        self.should_stop = state_dict['should_stop']


class ModelEMA:
    """
    模型参数的指数移动平均 (Exponential Moving Average)

    在训练过程中维护一个影子模型，其参数是训练模型参数的EMA。
    推理时使用EMA模型通常比直接使用训练模型效果更好。

    Args:
        model: 训练模型
        decay: EMA衰减率，通常设为0.9999
        tau: 衰减率的warmup参数，用于在训练初期使用较小的decay
        updates: 初始更新次数（用于恢复训练）
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # 创建EMA模型（深拷贝）
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = decay
        self.tau = tau

        # 冻结EMA模型参数
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """更新EMA模型参数"""
        self.updates += 1

        # 计算当前的decay值（带warmup）
        # 在训练初期使用较小的decay，让EMA模型快速跟上训练模型
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        # 获取模型参数
        msd = model.state_dict()
        esd = self.ema.state_dict()

        # 更新EMA参数
        for k, v in esd.items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """更新模型属性（如类别名称等）"""
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

    def state_dict(self):
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates
        }

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict['ema'])
        self.updates = state_dict['updates']


class GradientAccumulator:
    """
    梯度累积器

    当显存不足以使用大batch时，可以通过梯度累积模拟大batch训练。

    Args:
        accumulation_steps: 梯度累积步数

    Example:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for batch_idx, (images, targets) in enumerate(dataloader):
            loss = model(images, targets)
            accumulator.backward(loss)

            if accumulator.should_step(batch_idx):
                accumulator.step(optimizer)
                optimizer.zero_grad()
    """

    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def backward(self, loss):
        """执行反向传播，自动缩放loss"""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self.current_step += 1

    def should_step(self, batch_idx):
        """判断是否应该执行优化器step"""
        return (batch_idx + 1) % self.accumulation_steps == 0

    def step(self, optimizer, scaler=None, grad_clip=0.0, model=None):
        """
        执行优化器更新

        Args:
            optimizer: 优化器
            scaler: GradScaler（用于混合精度训练）
            grad_clip: 梯度裁剪阈值
            model: 模型（用于梯度裁剪）
        """
        if scaler is not None:
            # 混合精度训练
            if grad_clip > 0 and model is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通训练
            if grad_clip > 0 and model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        self.current_step = 0

    def get_effective_batch_size(self, batch_size):
        """获取有效batch size"""
        return batch_size * self.accumulation_steps


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失

    将one-hot标签平滑化，防止模型过于自信，提高泛化能力。

    Args:
        num_classes: 类别数
        smoothing: 平滑系数，通常设为0.1
        reduction: 损失的reduction方式
    """

    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测，shape (N, C) 或 (N, C, ...)
            target: 目标标签，shape (N,) 或 (N, C, ...)
        """
        # 如果target是类别索引，转换为one-hot
        if target.dim() == 1 or (target.dim() > 1 and target.size(-1) != self.num_classes):
            # target是类别索引
            with torch.no_grad():
                smooth_target = torch.zeros_like(pred)
                smooth_target.fill_(self.smoothing / (self.num_classes - 1))

                if target.dim() == 1:
                    smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
                else:
                    # 展平处理
                    original_shape = target.shape
                    target_flat = target.view(-1)
                    smooth_target_flat = smooth_target.view(-1, self.num_classes)
                    smooth_target_flat.scatter_(1, target_flat.unsqueeze(1), self.confidence)
                    smooth_target = smooth_target_flat.view(*original_shape, self.num_classes)
        else:
            # target已经是概率分布（soft labels）
            smooth_target = target * self.confidence + self.smoothing / self.num_classes

        # 计算交叉熵
        log_probs = torch.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCE:
    """
    用于二分类（BCE）的标签平滑

    将 0/1 标签平滑为 smoothing 和 1-smoothing

    Args:
        smoothing: 平滑系数，通常设为0.1
    """

    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.pos_label = 1.0 - smoothing
        self.neg_label = smoothing

    def smooth(self, targets):
        """平滑目标标签"""
        with torch.no_grad():
            return targets * self.pos_label + (1 - targets) * self.neg_label


class WarmupScheduler:
    """
    学习率Warmup调度器

    在训练初期线性增加学习率，然后使用指定的调度策略。

    Args:
        optimizer: 优化器
        warmup_epochs: warmup的epoch数
        warmup_bias_lr: bias参数的warmup起始学习率
        warmup_momentum: warmup期间的动量值
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs=3,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum

        # 记录初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.base_momentum = optimizer.param_groups[0].get('momentum', 0.9)

    def step(self, epoch, batch_idx=0, num_batches=1):
        """
        更新学习率

        Args:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            num_batches: 每个epoch的batch数
        """
        if epoch >= self.warmup_epochs:
            return

        # 计算warmup进度
        progress = (epoch * num_batches + batch_idx) / (self.warmup_epochs * num_batches)

        for i, group in enumerate(self.optimizer.param_groups):
            # 线性warmup学习率
            group['lr'] = self.base_lrs[i] * progress

            # 如果有momentum，也进行warmup
            if 'momentum' in group:
                group['momentum'] = self.warmup_momentum + (self.base_momentum - self.warmup_momentum) * progress
