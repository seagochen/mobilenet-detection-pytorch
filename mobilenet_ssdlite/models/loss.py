"""
YOLO Loss Functions for object detection
Fixed: Separate positive/negative objectness loss to handle class imbalance
Improved: CIoU loss for better box regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=False, CIoU=True, eps=1e-7):
    """
    Calculate IoU/CIoU between two sets of boxes.

    Args:
        box1: [N, 4] predicted boxes
        box2: [N, 4] target boxes
        xywh: if True, boxes are in (cx, cy, w, h) format; else (x1, y1, x2, y2)
        CIoU: if True, compute Complete IoU; else compute standard IoU

    Returns:
        iou: [N] IoU/CIoU values
    """
    if xywh:
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU:
        # Convex (smallest enclosing box) diagonal squared
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared

        # Center distance squared
        rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 +
                (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

        # Aspect ratio consistency
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + 1 + eps)

        return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


class YOLOLoss(nn.Module):
    """
    YOLO-style loss function
    Combines box regression, objectness, and classification losses

    Key fix: Separate obj/noobj loss to prevent background overwhelming foreground
    """

    def __init__(self, num_classes, anchors, strides, input_size, loss_weights=None, label_smoothing=0.0):
        """
        Args:
            num_classes: Number of object classes
            anchors: Anchor generator
            strides: Feature map strides [8, 16, 32]
            input_size: Input image size [H, W]
            loss_weights: Dict with 'box', 'obj', 'cls' weights
            label_smoothing: Label smoothing factor (0.0 = no smoothing, typical: 0.05-0.1)
        """
        super().__init__()

        self.num_classes = num_classes
        self.anchors = anchors
        self.strides = strides
        self.input_size = input_size
        self.label_smoothing = label_smoothing

        # Loss weights
        # Updated weights: increase box weight since CIoU values are typically smaller
        # Increase obj weight to force model to learn background suppression
        if loss_weights is None:
            loss_weights = {'box': 0.05, 'obj': 1.0, 'cls': 0.5}
        self.box_weight = loss_weights.get('box', 0.05) * 3.0  # 0.15 default
        self.obj_weight = loss_weights.get('obj', 1.0) * 5.0   # 5.0 default
        self.cls_weight = loss_weights.get('cls', 0.5)

        # NoObj weight: reduce background loss contribution to prevent
        # the massive number of negative samples from dominating training
        self.noobj_weight = 0.5

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, predictions, anchors, targets):
        """
        Compute loss

        Args:
            predictions: List of predictions for each scale
                        Each: [B, num_anchors, H, W, num_outputs]
            anchors: List of anchor boxes for each scale
            targets: List of target dicts, each containing:
                    - 'boxes': [N, 4] (x1, y1, x2, y2) in pixels
                    - 'labels': [N] class indices

        Returns:
            loss_dict: Dictionary with individual losses
            total_loss: Combined loss
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_num_pos = 0

        # Process each scale
        for scale_idx, (pred, anchor_boxes, stride) in enumerate(
            zip(predictions, anchors, self.strides)
        ):
            _, num_anchors, h, w, num_outputs = pred.shape

            # Split predictions
            box_pred = pred[..., :4]  # [B, num_anchors, H, W, 4]
            obj_pred = pred[..., 4]   # [B, num_anchors, H, W]
            cls_pred = pred[..., 5:]  # [B, num_anchors, H, W, num_classes]

            # Build targets for this scale
            # Use float32 for targets to avoid dtype mismatch with AMP
            obj_target = torch.zeros(obj_pred.shape, dtype=torch.float32, device=device)
            cls_target = torch.zeros(cls_pred.shape, dtype=torch.float32, device=device)

            # For CIoU loss: store GT boxes and anchor boxes for positive samples
            all_gt_boxes = []  # GT boxes in xyxy format
            all_anchor_boxes = []  # Anchor boxes in cxcywh format
            all_pos_indices = []  # (batch_idx, anchor_idx, gy, gx) tuples

            # Process each image in batch
            for batch_idx in range(batch_size):
                if targets[batch_idx] is None or len(targets[batch_idx]['boxes']) == 0:
                    continue

                gt_boxes = targets[batch_idx]['boxes'].to(device)  # [N, 4]
                gt_labels = targets[batch_idx]['labels'].to(device)  # [N]

                # Assign targets to anchors
                assigned_mask, assigned_anchors, assigned_targets, assigned_labels = \
                    self._assign_targets(
                        gt_boxes, gt_labels, anchor_boxes, stride, h, w
                    )

                if assigned_mask.sum() > 0:
                    # Get assigned positions
                    anchor_idx = assigned_mask.nonzero(as_tuple=True)

                    # Store GT boxes and anchors for CIoU computation
                    all_gt_boxes.append(assigned_targets)  # [M, 4] xyxy
                    all_anchor_boxes.append(assigned_anchors)  # [M, 4] cxcywh

                    # Store indices for extracting predictions
                    for i in range(len(assigned_targets)):
                        all_pos_indices.append((
                            batch_idx,
                            anchor_idx[0][i].item(),
                            anchor_idx[1][i].item(),
                            anchor_idx[2][i].item()
                        ))

                    # Objectness target
                    obj_target[batch_idx][anchor_idx] = 1.0

                    # Class targets (one-hot with optional label smoothing)
                    one_hot = F.one_hot(
                        assigned_labels.long(),
                        num_classes=self.num_classes
                    ).float()
                    # Apply label smoothing: smooth = (1 - eps) * one_hot + eps / num_classes
                    if self.label_smoothing > 0:
                        one_hot = one_hot * (1.0 - self.label_smoothing) + \
                                  self.label_smoothing / self.num_classes
                    cls_target[batch_idx][anchor_idx] = one_hot

            # Compute losses
            pos_mask = obj_target > 0
            neg_mask = ~pos_mask
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()

            # Box loss using CIoU (only for positive samples)
            if num_pos > 0 and len(all_gt_boxes) > 0:
                # Concatenate all GT and anchor boxes
                gt_boxes_cat = torch.cat(all_gt_boxes, dim=0)  # [num_pos, 4] xyxy
                anchor_boxes_cat = torch.cat(all_anchor_boxes, dim=0)  # [num_pos, 4] cxcywh

                # Extract predictions for positive samples and decode to xyxy
                pred_offsets = []
                for (b, a, gy, gx) in all_pos_indices:
                    pred_offsets.append(box_pred[b, a, gy, gx])
                pred_offsets = torch.stack(pred_offsets, dim=0)  # [num_pos, 4]

                # Decode predictions: offset -> xyxy coordinates
                pred_boxes = self._decode_boxes(pred_offsets, anchor_boxes_cat, stride)

                # Compute CIoU loss
                ciou = bbox_iou(pred_boxes, gt_boxes_cat, xywh=False, CIoU=True)
                box_loss = (1.0 - ciou).mean()
                total_box_loss += box_loss
                total_num_pos += num_pos

            # === KEY FIX: Separate Obj/NoObj Loss ===
            # This prevents the massive number of background samples from
            # overwhelming the few positive samples

            # Positive objectness loss (foreground)
            if num_pos > 0:
                obj_loss_pos = self.bce_loss(
                    obj_pred[pos_mask],
                    obj_target[pos_mask]
                ).mean()
            else:
                obj_loss_pos = torch.tensor(0.0, device=device)

            # Negative objectness loss (background) - weighted down
            if num_neg > 0:
                obj_loss_neg = self.bce_loss(
                    obj_pred[neg_mask],
                    obj_target[neg_mask]
                ).mean() * self.noobj_weight
            else:
                obj_loss_neg = torch.tensor(0.0, device=device)

            total_obj_loss += obj_loss_pos + obj_loss_neg

            # Class loss (only for positive samples)
            if num_pos > 0:
                cls_loss = self.bce_loss(cls_pred[pos_mask], cls_target[pos_mask]).mean()
                total_cls_loss += cls_loss

        # Average over scales
        num_scales = len(predictions)
        total_box_loss = total_box_loss / num_scales if total_num_pos > 0 else torch.tensor(0.0, device=device)
        total_obj_loss = total_obj_loss / num_scales
        total_cls_loss = total_cls_loss / num_scales if total_num_pos > 0 else torch.tensor(0.0, device=device)

        # Weighted sum
        total_loss = (
            self.box_weight * total_box_loss +
            self.obj_weight * total_obj_loss +
            self.cls_weight * total_cls_loss
        )

        loss_dict = {
            'box_loss': total_box_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'cls_loss': total_cls_loss.item(),
            'total_loss': total_loss.item()
        }

        return loss_dict, total_loss

    def _assign_targets(self, gt_boxes, gt_labels, anchors, stride, h, w):
        """
        Assign ground truth boxes to anchors using IoU matching

        Args:
            gt_boxes: [N, 4] ground truth boxes (x1, y1, x2, y2)
            gt_labels: [N] class labels
            anchors: [num_anchors, H, W, 4] anchor boxes
            stride: Feature map stride
            h, w: Feature map height and width

        Returns:
            assigned_mask: [num_anchors, H, W] bool mask
            assigned_anchors: [M, 4] assigned anchor boxes
            assigned_targets: [M, 4] assigned target boxes
            assigned_labels: [M] assigned labels
        """
        device = gt_boxes.device
        num_anchors = anchors.size(0)

        # Convert gt_boxes to center format
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

        # Find which grid cell each GT box falls into
        gt_grid_x = (gt_cx / stride).long().clamp(0, w - 1)
        gt_grid_y = (gt_cy / stride).long().clamp(0, h - 1)

        # Use dict to track assignments, keeping highest IoU for each position
        # Key: (anchor_idx, gy, gx), Value: (iou, gt_idx)
        assignments = {}

        # IoU threshold for multi-anchor matching
        # Anchors with IoU > threshold will all be assigned as positive samples
        iou_threshold = 0.5

        for i in range(len(gt_boxes)):
            gx, gy = gt_grid_x[i].item(), gt_grid_y[i].item()

            # Get anchors at this grid cell
            cell_anchors = anchors[:, gy, gx, :]  # [num_anchors, 4]

            # Compute IoU with all anchors at this location
            ious = self._compute_iou(
                gt_boxes[i:i+1],
                cell_anchors
            ).squeeze(0)  # [num_anchors]

            # Multi-anchor matching: assign all anchors with IoU > threshold
            # If no anchor exceeds threshold, assign the best one (fallback)
            positive_mask = ious > iou_threshold
            if positive_mask.sum() == 0:
                # Fallback: at least assign the best anchor
                positive_mask[ious.argmax()] = True

            # Assign all positive anchors
            for anchor_idx in positive_mask.nonzero(as_tuple=True)[0]:
                anchor_idx = anchor_idx.item()
                iou_val = ious[anchor_idx].item()

                key = (anchor_idx, gy, gx)
                # Only update if this GT has higher IoU than existing assignment
                if key not in assignments or iou_val > assignments[key][0]:
                    assignments[key] = (iou_val, i)

        # Build output tensors from assignments
        assigned_mask = torch.zeros(num_anchors, h, w, dtype=torch.bool, device=device)
        assigned_anchors_list = []
        assigned_targets_list = []
        assigned_labels_list = []

        for (anchor_idx, gy, gx), (_, gt_idx) in assignments.items():
            assigned_mask[anchor_idx, gy, gx] = True
            assigned_anchors_list.append(anchors[anchor_idx, gy, gx, :])
            assigned_targets_list.append(gt_boxes[gt_idx])
            assigned_labels_list.append(gt_labels[gt_idx])

        if len(assigned_anchors_list) > 0:
            assigned_anchors = torch.stack(assigned_anchors_list)
            assigned_targets = torch.stack(assigned_targets_list)
            assigned_labels = torch.stack(assigned_labels_list)
        else:
            assigned_anchors = torch.zeros(0, 4, device=device)
            assigned_targets = torch.zeros(0, 4, device=device)
            assigned_labels = torch.zeros(0, dtype=torch.long, device=device)

        return assigned_mask, assigned_anchors, assigned_targets, assigned_labels

    def _compute_box_targets(self, gt_boxes, anchors, stride):
        """
        Compute box regression targets (offsets from anchors)

        Args:
            gt_boxes: [N, 4] target boxes (x1, y1, x2, y2)
            anchors: [N, 4] anchor boxes (cx, cy, w, h)
            stride: Current feature map stride

        Returns:
            targets: [N, 4] regression targets (tx, ty, tw, th)
        """
        # Convert gt_boxes to center format
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

        # Get anchor parameters
        anchor_cx = anchors[:, 0]
        anchor_cy = anchors[:, 1]
        anchor_w = anchors[:, 2]
        anchor_h = anchors[:, 3]

        # Compute targets (inverse of decoding)
        # tx, ty: normalized offset using current stride
        tx = (gt_cx - anchor_cx) / stride
        ty = (gt_cy - anchor_cy) / stride

        # tw, th: log-space scaling
        tw = torch.log(gt_w / anchor_w + 1e-16)
        th = torch.log(gt_h / anchor_h + 1e-16)

        targets = torch.stack([tx, ty, tw, th], dim=-1)
        return targets

    def _decode_boxes(self, offsets, anchors, stride):
        """
        Decode box predictions from offsets to xyxy coordinates.
        IMPORTANT: Must match inference decoder logic in detection_head.py

        Args:
            offsets: [N, 4] raw network outputs (tx, ty, tw, th)
            anchors: [N, 4] anchor boxes (cx, cy, w, h)
            stride: Current feature map stride

        Returns:
            boxes: [N, 4] decoded boxes (x1, y1, x2, y2)
        """
        # Get anchor parameters
        anchor_cx = anchors[:, 0]
        anchor_cy = anchors[:, 1]
        anchor_w = anchors[:, 2]
        anchor_h = anchors[:, 3]

        # Decode offsets - MUST match detection_head.py YOLODecoder
        # Decode center: Apply sigmoid and scale to [-0.5, 1.5] range (YOLOv5 style)
        pred_cx = (torch.sigmoid(offsets[:, 0]) - 0.5) * 2 * stride + anchor_cx
        pred_cy = (torch.sigmoid(offsets[:, 1]) - 0.5) * 2 * stride + anchor_cy

        # Decode size using YOLOv5/v8 style: (sigmoid(x)*2)^2
        # MUST match detection_head.py YOLODecoder exactly
        pred_w = (torch.sigmoid(offsets[:, 2]) * 2) ** 2 * anchor_w
        pred_h = (torch.sigmoid(offsets[:, 3]) * 2) ** 2 * anchor_h

        # Convert to xyxy
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _compute_iou(boxes1, boxes2):
        """
        Compute IoU between boxes

        Args:
            boxes1: [N, 4] (x1, y1, x2, y2)
            boxes2: [M, 4] (cx, cy, w, h) or (x1, y1, x2, y2)

        Returns:
            iou: [N, M]
        """
        # Convert boxes2 to (x1, y1, x2, y2) if in center format
        if boxes2.shape[-1] == 4:
            # Assume (cx, cy, w, h)
            boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
            boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
            boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
            boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
            boxes2_xyxy = torch.stack([boxes2_x1, boxes2_y1, boxes2_x2, boxes2_y2], dim=-1)
        else:
            boxes2_xyxy = boxes2

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

        # Intersection
        lt = torch.max(boxes1[:, None, :2], boxes2_xyxy[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2_xyxy[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        # Union
        union = area1[:, None] + area2 - inter

        iou = inter / (union + 1e-16)
        return iou
