"""
梯度累积模块
"""

import torch


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
