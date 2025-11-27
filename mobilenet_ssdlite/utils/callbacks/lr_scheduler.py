"""
学习率调度器模块
包含：ReduceLROnPlateau, WarmupScheduler
"""


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
