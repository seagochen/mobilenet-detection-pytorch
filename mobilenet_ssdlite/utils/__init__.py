"""
Utility modules for MobileNet-SSDLite object detection.

Subpackages:
- data: Dataset loading, transforms, image preprocessing, anchors
- ops: Box operations, general utilities
- metrics: Evaluation metrics (mAP, confusion matrix)
- visualization: Plotting and visualization
"""

# =============================================================================
# Data utilities
# =============================================================================
from .data import (
    # Dataset
    DetectionDataset, YOLODataset, collate_fn,
    # Transforms
    get_transforms,
    # Image preprocessing
    IMAGENET_MEAN, IMAGENET_STD, get_normalize_params,
    normalize, denormalize, denormalize_to_uint8,
    # Path utilities
    resolve_split_path, infer_label_dir,
    load_yolo_yaml, find_image_files,
    get_label_path, validate_dataset_structure,
    # Anchor utilities
    get_or_compute_anchors, compute_anchors_for_dataset, kmeans_anchors,
    save_anchors, load_anchors, load_boxes_from_yolo_dataset,
)

# =============================================================================
# Box operations and general utilities
# =============================================================================
from .ops import (
    # Box operations
    box_iou, box_iou_pairwise, box_ciou,
    box_iou_numpy, box_iou_wh, box_iou_single,
    nms, batched_nms,
    xyxy_to_xywh, xywh_to_xyxy,
    clip_boxes, remove_small_boxes,
    # General utilities
    check_img_size, init_seeds, check_file, increment_path, colorstr,
)

# =============================================================================
# Metrics
# =============================================================================
from .metrics import (
    DetectionMetrics, ConfusionMatrix,
    box_iou_batch, ap_per_class, compute_ap,
)

# =============================================================================
# Visualization
# =============================================================================
from .visualization import (
    TrainingPlotter, plot_detection_samples, plot_labels_distribution,
    visualize_detections, get_color_palette, plot_training_curves,
)

# =============================================================================
# Training callbacks
# =============================================================================
from .callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelEMA, GradientAccumulator,
    LabelSmoothingLoss, LabelSmoothingBCE, WarmupScheduler
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
    'nms', 'batched_nms',
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
