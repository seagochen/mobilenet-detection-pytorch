"""
Data loading and preprocessing utilities.

This subpackage contains:
- Dataset classes (COCO, YOLO formats)
- Data augmentation transforms
- Image preprocessing utilities
- Path resolution utilities
"""

from .dataset import DetectionDataset, YOLODataset, collate_fn
from .transforms import get_transforms
from .image import (
    IMAGENET_MEAN, IMAGENET_STD, get_normalize_params,
    normalize, denormalize, denormalize_to_uint8
)
from .path_utils import resolve_split_path, infer_label_dir
from .anchors import (
    get_or_compute_anchors, compute_anchors_for_dataset, kmeans_anchors,
    save_anchors, load_anchors, load_boxes_from_yolo_dataset, compute_pos_neg_ratio
)

__all__ = [
    # Dataset
    'DetectionDataset', 'YOLODataset', 'collate_fn',
    # Transforms
    'get_transforms',
    # Image preprocessing
    'IMAGENET_MEAN', 'IMAGENET_STD', 'get_normalize_params',
    'normalize', 'denormalize', 'denormalize_to_uint8',
    # Path utilities
    'resolve_split_path', 'infer_label_dir',
    # Anchor utilities
    'get_or_compute_anchors', 'compute_anchors_for_dataset', 'kmeans_anchors',
    'save_anchors', 'load_anchors', 'load_boxes_from_yolo_dataset', 'compute_pos_neg_ratio',
]
