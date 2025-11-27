"""
Unified bounding box operations.
Consolidates IoU computation, NMS, and box format conversions.
"""
import math
import numpy as np
import torch
import torchvision
from typing import Tuple, Optional, Union


# =============================================================================
# Box Format Conversions
# =============================================================================

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: [..., 4] tensor in xyxy format

    Returns:
        boxes: [..., 4] tensor in xywh (center) format
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: [..., 4] tensor in xywh (center) format

    Returns:
        boxes: [..., 4] tensor in xyxy format
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# =============================================================================
# IoU Computation - PyTorch (for training/inference)
# =============================================================================

def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes (vectorized).

    Args:
        boxes1: [N, 4] tensor in (x1, y1, x2, y2) format
        boxes2: [M, 4] tensor in (x1, y1, x2, y2) format
        eps: small value for numerical stability

    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2 - inter + eps

    return inter / union


def box_iou_pairwise(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    xywh: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute IoU between corresponding pairs of boxes (element-wise).

    Args:
        boxes1: [N, 4] tensor
        boxes2: [N, 4] tensor (same N as boxes1)
        xywh: if True, boxes are in (cx, cy, w, h) format; else (x1, y1, x2, y2)
        eps: small value for numerical stability

    Returns:
        iou: [N] IoU values for each pair
    """
    if xywh:
        boxes1 = xywh_to_xyxy(boxes1)
        boxes2 = xywh_to_xyxy(boxes2)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter + eps

    return inter / union


def box_ciou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    xywh: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU) between corresponding pairs of boxes.
    CIoU = IoU - (distance^2 / c^2) - alpha * v
    where v measures aspect ratio consistency.

    Args:
        boxes1: [N, 4] predicted boxes
        boxes2: [N, 4] target boxes
        xywh: if True, boxes are in (cx, cy, w, h) format; else (x1, y1, x2, y2)
        eps: small value for numerical stability

    Returns:
        ciou: [N] CIoU values (range [-1, 1])
    """
    if xywh:
        boxes1 = xywh_to_xyxy(boxes1)
        boxes2 = xywh_to_xyxy(boxes2)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # Convex (smallest enclosing box) diagonal squared
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance squared
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 +
            (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

    # Aspect ratio consistency
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    # CIoU
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou.clamp(min=-1.0, max=1.0)


# =============================================================================
# IoU Computation - NumPy (for evaluation/metrics)
# =============================================================================

def box_iou_numpy(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    eps: float = 1e-16
) -> np.ndarray:
    """
    Compute IoU between two sets of boxes (NumPy version for evaluation).

    Args:
        boxes1: [N, 4] array in (x1, y1, x2, y2) format
        boxes2: [M, 4] array in (x1, y1, x2, y2) format
        eps: small value for numerical stability

    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    x1 = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0][None, :])
    y1 = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1][None, :])
    x2 = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2][None, :])
    y2 = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3][None, :])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    union = area1[:, None] + area2[None, :] - inter + eps

    return inter / union


def box_iou_wh(
    boxes_wh: np.ndarray,
    anchors_wh: np.ndarray,
    eps: float = 1e-16
) -> np.ndarray:
    """
    Compute IoU between boxes and anchors using only width/height
    (assumes centers are aligned). Used for anchor clustering.

    Args:
        boxes_wh: [N, 2] box widths and heights
        anchors_wh: [K, 2] anchor widths and heights
        eps: small value for numerical stability

    Returns:
        iou: [N, K] IoU matrix
    """
    # Expand dimensions for broadcasting
    boxes = boxes_wh[:, np.newaxis, :]  # [N, 1, 2]
    anchors = anchors_wh[np.newaxis, :, :]  # [1, K, 2]

    # Intersection (assuming centers are aligned)
    inter_w = np.minimum(boxes[..., 0], anchors[..., 0])
    inter_h = np.minimum(boxes[..., 1], anchors[..., 1])
    inter_area = inter_w * inter_h

    # Union
    boxes_area = boxes[..., 0] * boxes[..., 1]
    anchors_area = anchors[..., 0] * anchors[..., 1]
    union_area = boxes_area + anchors_area - inter_area + eps

    return inter_area / union_area


# =============================================================================
# Non-Maximum Suppression
# =============================================================================

def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    class_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
        class_ids: [N] optional class IDs for class-aware NMS

    Returns:
        keep: indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    if class_ids is not None:
        # Class-aware NMS using coordinate offset trick
        max_coordinate = boxes.max() + 1
        offsets = class_ids.float() * max_coordinate
        boxes_for_nms = boxes + offsets[:, None]
    else:
        boxes_for_nms = boxes

    return torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Apply batched NMS (class-aware NMS using torchvision).

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        class_ids: [N] class IDs
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: indices of boxes to keep
    """
    return torchvision.ops.batched_nms(boxes, scores, class_ids, iou_threshold)


# =============================================================================
# Box Clipping and Validation
# =============================================================================

def clip_boxes(
    boxes: torch.Tensor,
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Clip boxes to image boundaries.

    Args:
        boxes: [..., 4] boxes in (x1, y1, x2, y2) format
        image_size: (height, width) of the image

    Returns:
        clipped_boxes: boxes clipped to image boundaries
    """
    h, w = image_size
    boxes[..., 0] = boxes[..., 0].clamp(min=0, max=w)  # x1
    boxes[..., 1] = boxes[..., 1].clamp(min=0, max=h)  # y1
    boxes[..., 2] = boxes[..., 2].clamp(min=0, max=w)  # x2
    boxes[..., 3] = boxes[..., 3].clamp(min=0, max=h)  # y2
    return boxes


def remove_small_boxes(
    boxes: torch.Tensor,
    min_size: float = 1.0
) -> torch.Tensor:
    """
    Remove boxes with width or height smaller than min_size.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        min_size: minimum box dimension

    Returns:
        keep: boolean mask of valid boxes
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return (w >= min_size) & (h >= min_size)


# =============================================================================
# Legacy Compatibility
# =============================================================================

def box_iou_single(box1: Tuple, box2: Tuple, eps: float = 1e-16) -> float:
    """
    Compute IoU between two boxes (legacy single-pair interface).

    Args:
        box1, box2: (x1, y1, x2, y2) tuples
        eps: small value for numerical stability

    Returns:
        iou: IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area + eps

    return inter_area / union_area


# Aliases for backward compatibility
bbox_iou = box_ciou  # loss.py used bbox_iou with CIoU=True as default
box_iou_batch = box_iou_numpy  # metrics.py used this name
