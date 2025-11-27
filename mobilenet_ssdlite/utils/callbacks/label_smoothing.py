"""
标签平滑模块
"""

import torch
import torch.nn as nn


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
