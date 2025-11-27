"""
训练回调函数模块

包含：
- 学习率调度: ReduceLROnPlateau, WarmupScheduler
- 早停机制: EarlyStopping
- 模型EMA: ModelEMA
- 梯度累积: GradientAccumulator
- 标签平滑: LabelSmoothingLoss, LabelSmoothingBCE
"""

from .lr_scheduler import ReduceLROnPlateau, WarmupScheduler
from .early_stopping import EarlyStopping
from .ema import ModelEMA
from .gradient import GradientAccumulator
from .label_smoothing import LabelSmoothingLoss, LabelSmoothingBCE

__all__ = [
    'ReduceLROnPlateau',
    'WarmupScheduler',
    'EarlyStopping',
    'ModelEMA',
    'GradientAccumulator',
    'LabelSmoothingLoss',
    'LabelSmoothingBCE',
]
