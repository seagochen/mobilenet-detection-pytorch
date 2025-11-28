"""
Evaluation metrics for object detection.

This subpackage contains:
- mAP computation
- Precision/Recall curves
"""

from .metrics import DetectionMetrics, ap_per_class, compute_ap

__all__ = [
    'DetectionMetrics',
    'ap_per_class', 'compute_ap',
]
