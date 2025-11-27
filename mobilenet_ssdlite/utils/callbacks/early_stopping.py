"""
早停机制模块
"""


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

    def step(self, metrics=None, lr_reduced=False, num_lr_reductions=0, improved=False):
        """
        检查是否应该停止训练

        Args:
            metrics: 当前epoch的验证指标（可选，当 improved 参数提供时可以省略）
            lr_reduced: 本次是否降低了学习率（配合 check_lr_reductions）
            num_lr_reductions: 学习率降低总次数
            improved: 是否有改善（外部判断，如 best 模型被更新）。
                      如果为 True，计数器将被重置。这允许外部使用不同指标
                      （如 mAP）来判断改善，而不仅仅依赖传入的 metrics。

        Returns:
            bool: 是否应该停止训练
        """
        # 如果外部告知有改善，重置计数器
        if improved:
            self.counter = 0
            if metrics is not None:
                current = float(metrics)
                if self.best is None or self.is_better(current, self.best):
                    self.best = current
            return self.should_stop

        # 如果没有提供 metrics 且没有 improved，直接返回
        if metrics is None:
            return self.should_stop

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
