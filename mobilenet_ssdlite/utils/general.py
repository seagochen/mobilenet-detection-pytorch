"""
General utility functions.
Provides file operations, seeding, and terminal utilities.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# Re-export box operations from box_ops for backward compatibility
from .box_ops import box_iou_single as box_iou, nms as _nms_tensor


def nms(
    detections: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Non-Maximum Suppression for detection results.

    Args:
        detections: List of detection dicts with 'class_id', 'confidence', 'bbox'
        iou_threshold: IoU threshold for suppression

    Returns:
        filtered: Filtered detection results after NMS
    """
    if len(detections) == 0:
        return []

    # Extract data to tensors
    boxes = torch.tensor([det['bbox'] for det in detections], dtype=torch.float32)
    scores = torch.tensor([det['confidence'] for det in detections], dtype=torch.float32)
    class_ids = torch.tensor([det['class_id'] for det in detections], dtype=torch.int64)

    # Apply NMS using box_ops
    keep_indices = _nms_tensor(boxes, scores, iou_threshold, class_ids)

    # Rebuild result list
    return [detections[idx] for idx in keep_indices.numpy()]


def check_img_size(img_size: int, stride: int = 32) -> int:
    """
    验证img_size是stride的倍数

    Args:
        img_size: 图像尺寸
        stride: 下采样步长

    Returns:
        验证后的img_size
    """
    new_size = max(stride, (img_size // stride) * stride)
    if new_size != img_size:
        print(f'Warning: img_size {img_size} is not a multiple of stride {stride}. '
              f'Using {new_size} instead.')
    return new_size


def init_seeds(seed: int = 0):
    """
    初始化随机种子以确保可复现性

    Args:
        seed: 随机种子
    """
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保可复现性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_file(file: str) -> Path:
    """
    验证文件是否存在

    Args:
        file: 文件路径

    Returns:
        Path对象
    """
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f'File not found: {file}')
    return file


def increment_path(path: Path, exist_ok: bool = False, sep: str = '_') -> Path:
    """
    自动增加路径后缀以避免覆盖

    Args:
        path: 路径
        exist_ok: 如果为True，允许路径存在
        sep: 分隔符

    Returns:
        新路径

    Example:
        runs/exp -> runs/exp_2 -> runs/exp_3
    """
    path = Path(path)

    if not path.exists() or exist_ok:
        return path

    path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

    # 查找下一个可用编号
    for n in range(2, 100):
        p = f'{path}{sep}{n}{suffix}'
        if not Path(p).exists():
            return Path(p)

    # 如果都存在，使用时间戳
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return Path(f'{path}{sep}{timestamp}{suffix}')


def colorstr(*args):
    """
    彩色字符串（用于终端输出）

    Example:
        colorstr('blue', 'bold', 'hello')
    """
    *args, string = args if len(args) > 1 else ('blue', 'bold', args[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }

    return ''.join(colors.get(x, '') for x in args) + f'{string}' + colors['end']


if __name__ == '__main__':
    print("Testing general utilities...")

    # Test NMS
    detections = [
        {'class_id': 0, 'confidence': 0.9, 'bbox': (10, 10, 50, 50)},
        {'class_id': 0, 'confidence': 0.8, 'bbox': (15, 15, 55, 55)},
        {'class_id': 1, 'confidence': 0.7, 'bbox': (100, 100, 150, 150)},
    ]
    filtered = nms(detections, iou_threshold=0.5)
    print(f"NMS: {len(detections)} -> {len(filtered)} detections")

    # Test IoU
    iou = box_iou((10, 10, 50, 50), (15, 15, 55, 55))
    print(f"IoU: {iou:.4f}")

    # Test img_size check
    size = check_img_size(639, stride=32)
    print(f"Img size: 639 -> {size}")

    # Test colorstr
    print(colorstr('blue', 'bold', 'This is a test'))

    print("\n✓ All tests passed!")
