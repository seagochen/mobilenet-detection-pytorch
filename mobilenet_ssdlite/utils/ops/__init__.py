"""
Box operations and general utilities.

This subpackage contains:
- Box operations (IoU, NMS, format conversion)
- General utilities (seed init, path handling, etc.)
"""

from .box_ops import (
    box_iou, box_iou_pairwise, box_ciou,
    box_iou_numpy, box_iou_wh,
    nms,
    xyxy_to_xywh, xywh_to_xyxy,
)

from .general import (
    check_img_size, init_seeds, check_file, increment_path, colorstr
)

__all__ = [
    # Box operations
    'box_iou', 'box_iou_pairwise', 'box_ciou',
    'box_iou_numpy', 'box_iou_wh',
    'nms',
    'xyxy_to_xywh', 'xywh_to_xyxy',
    # General utilities
    'check_img_size', 'init_seeds', 'check_file', 'increment_path', 'colorstr',
]
