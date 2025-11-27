"""
Evaluation metrics for object detection.

This subpackage contains:
- mAP computation
- Precision/Recall curves
- Confusion matrix
"""

from .metrics import (
    DetectionMetrics, ConfusionMatrix,
    ap_per_class, compute_ap
)

# Re-export box_iou_batch for backward compatibility
from ..ops.box_ops import box_iou_numpy as box_iou_batch

__all__ = [
    'DetectionMetrics', 'ConfusionMatrix',
    'box_iou_batch', 'ap_per_class', 'compute_ap',
]
