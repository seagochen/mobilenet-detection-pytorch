"""
Utility modules for MobileNet-SSDLite object detection.
"""

# Dataset utilities
from .dataset import DetectionDataset, YOLODataset, collate_fn
from .transforms import get_transforms

# Image preprocessing
from .image import (
    IMAGENET_MEAN, IMAGENET_STD, get_normalize_params,
    normalize, denormalize, denormalize_to_uint8
)

# Box operations (unified module)
from .box_ops import (
    box_iou, box_iou_pairwise, box_ciou,
    box_iou_numpy, box_iou_wh,
    nms, batched_nms,
    xyxy_to_xywh, xywh_to_xyxy,
    clip_boxes, remove_small_boxes,
    box_iou_single,  # Legacy single-pair IoU
)

# Path utilities
from .path_utils import (
    resolve_split_path, infer_label_dir,
    load_yolo_yaml, find_image_files,
    get_label_path, validate_dataset_structure
)

# Visualization (unified module)
from .plots import (
    TrainingPlotter, plot_detection_samples, plot_labels_distribution,
    visualize_detections, get_color_palette,
)
from .visualize import plot_training_curves  # Legacy alias

# Metrics
from .metrics import (
    DetectionMetrics, ConfusionMatrix,
    box_iou_batch, ap_per_class, compute_ap
)

# Training callbacks
from .callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelEMA, GradientAccumulator,
    LabelSmoothingLoss, LabelSmoothingBCE, WarmupScheduler
)

# General utilities
from .general import (
    check_img_size, init_seeds, check_file, increment_path, colorstr
)
# Legacy NMS wrapper (uses box_ops internally)
from .general import nms as nms_detections

# Anchor utilities
from .anchors import (
    get_or_compute_anchors, compute_anchors_for_dataset, kmeans_anchors,
    save_anchors, load_anchors, load_boxes_from_yolo_dataset
)

__all__ = [
    # Dataset
    'DetectionDataset', 'YOLODataset', 'collate_fn', 'get_transforms',

    # Image preprocessing
    'IMAGENET_MEAN', 'IMAGENET_STD', 'get_normalize_params',
    'normalize', 'denormalize', 'denormalize_to_uint8',

    # Box operations
    'box_iou', 'box_iou_pairwise', 'box_ciou',
    'box_iou_numpy', 'box_iou_wh', 'box_iou_batch', 'box_iou_single',
    'nms', 'batched_nms', 'nms_detections',
    'xyxy_to_xywh', 'xywh_to_xyxy',
    'clip_boxes', 'remove_small_boxes',

    # Path utilities
    'resolve_split_path', 'infer_label_dir',
    'load_yolo_yaml', 'find_image_files',
    'get_label_path', 'validate_dataset_structure',

    # Visualization
    'visualize_detections', 'TrainingPlotter',
    'plot_detection_samples', 'plot_labels_distribution',
    'plot_training_curves', 'get_color_palette',

    # Metrics
    'DetectionMetrics', 'ConfusionMatrix', 'ap_per_class', 'compute_ap',

    # Callbacks
    'ReduceLROnPlateau', 'EarlyStopping', 'ModelEMA', 'GradientAccumulator',
    'LabelSmoothingLoss', 'LabelSmoothingBCE', 'WarmupScheduler',

    # General utilities
    'check_img_size', 'init_seeds', 'check_file', 'increment_path', 'colorstr',

    # Anchor utilities
    'get_or_compute_anchors', 'compute_anchors_for_dataset', 'kmeans_anchors',
    'save_anchors', 'load_anchors', 'load_boxes_from_yolo_dataset',
]
