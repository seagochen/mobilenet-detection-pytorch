"""
模型指数移动平均 (EMA) 模块
"""

import math
from copy import deepcopy


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
