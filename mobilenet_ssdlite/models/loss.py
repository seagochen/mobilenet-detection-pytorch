"""
Detection Loss Functions for object detection.
Uses unified box operations and codec from shared modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mobilenet_ssdlite.utils import box_ciou, box_iou, xywh_to_xyxy
from .box_codec import YOLOBoxCodec


class DetectionLoss(nn.Module):
    """
    Anchor-based detection loss function.
    Combines box regression (CIoU), objectness, and classification losses.

    Key fix: Separate obj/noobj loss to prevent background overwhelming foreground.
    """

    def __init__(self, num_classes, anchors, strides, input_size, loss_weights=None,
                 label_smoothing=0.0):
        """
        Args:
            num_classes: Number of object classes
            anchors: Anchor generator
            strides: Feature map strides [8, 16, 32]
            input_size: Input image size [H, W]
            loss_weights: Dict with 'box', 'obj_pos', 'obj_neg', 'cls' weights
            label_smoothing: Label smoothing factor (0.0 = no smoothing, typical: 0.05-0.1)
        """
        super().__init__()

        self.num_classes = num_classes
        self.anchors = anchors
        self.strides = strides
        self.input_size = input_size
        self.label_smoothing = label_smoothing

        # Four-component loss weights:
        # total_loss = w_box * box + w_obj_pos * obj_pos + w_obj_neg * obj_neg + w_cls * cls
        # All losses are normalized (mean per sample), so weights are directly comparable
        if loss_weights is None:
            loss_weights = {}
        self.w_box = loss_weights.get('box', 1.0)
        self.w_obj_pos = loss_weights.get('obj_pos', 1.0)
        self.w_obj_neg = loss_weights.get('obj_neg', 1.0)
        self.w_cls = loss_weights.get('cls', 1.0)

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
        dtype = predictions[0].dtype
        batch_size = predictions[0].size(0)

        # Global accumulators for loss sums (not means)
        # This allows proper normalization at the end
        loss_items = {
            'box_sum': torch.zeros(1, device=device, dtype=dtype).squeeze(),
            'cls_sum': torch.zeros(1, device=device, dtype=dtype).squeeze(),
            'num_pos': 0,
            'num_anchors': 0,
        }
        # obj_pos_sum and obj_neg_sum will be added dynamically

        # Process each scale
        for scale_idx, (pred, anchor_boxes, stride) in enumerate(
            zip(predictions, anchors, self.strides)
        ):
            _, num_anchors, h, w, num_outputs = pred.shape
            loss_items['num_anchors'] += batch_size * num_anchors * h * w

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

                    # Objectness target (will be updated to IoU later)
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
            loss_items['num_pos'] += num_pos

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

                # Compute CIoU loss with numerical stability
                ciou = box_ciou(pred_boxes, gt_boxes_cat, xywh=False)

                # === IoU-Aware Objectness ===
                # Update objectness target to be CIoU value instead of hard 1.0
                # This teaches the model: "better box prediction = higher confidence"
                # Key technique from YOLOv5/v7/v8 for aligning classification with localization
                ciou_detached = ciou.detach().clamp(0, 1)
                for k, (b, a, gy, gx) in enumerate(all_pos_indices):
                    obj_target[b, a, gy, gx] = ciou_detached[k]

                # Accumulate box loss (sum, not mean)
                loss_items['box_sum'] = loss_items['box_sum'] + (1.0 - ciou).sum()

            # === Objectness Loss ===
            # Separate positive and negative objectness loss for better balance
            # This prevents the massive number of negative samples from dominating
            obj_loss_all = self.bce_loss(
                obj_pred.float(),  # Ensure float32 for stability
                obj_target
            )

            # Positive samples: sum their loss (will be normalized by num_pos later)
            if num_pos > 0:
                loss_items['obj_pos_sum'] = loss_items.get('obj_pos_sum', torch.tensor(0.0, device=device)) + obj_loss_all[pos_mask].sum()

            # Negative samples: compute mean per scale (will be averaged across scales later)
            num_neg = neg_mask.sum().item()
            if num_neg > 0:
                noobj_loss_mean = obj_loss_all[neg_mask].mean()
                loss_items['obj_neg_sum'] = loss_items.get('obj_neg_sum', torch.tensor(0.0, device=device)) + noobj_loss_mean

            # Class loss (only for positive samples)
            if num_pos > 0:
                cls_loss = self.bce_loss(cls_pred[pos_mask], cls_target[pos_mask])
                loss_items['cls_sum'] = loss_items['cls_sum'] + cls_loss.sum()

        # === Aggregate Losses ===
        # Simple and intuitive four-component loss:
        # total_loss = w_box * box + w_obj_pos * obj_pos + w_obj_neg * obj_neg + w_cls * cls
        total_pos = max(loss_items['num_pos'], 1)
        num_scales = len(predictions)

        # Each loss is normalized to be a "mean loss per sample"
        box_loss = loss_items['box_sum'] / total_pos           # mean box loss per positive sample
        obj_pos_loss = loss_items.get('obj_pos_sum', torch.tensor(0.0, device=device)) / total_pos  # mean obj loss per positive
        obj_neg_loss = loss_items.get('obj_neg_sum', torch.tensor(0.0, device=device)) / num_scales  # mean obj loss per scale (already mean per neg)
        cls_loss = loss_items['cls_sum'] / total_pos           # mean cls loss per positive sample

        # Four-component weighted sum
        total_loss = (
            self.w_box * box_loss +
            self.w_obj_pos * obj_pos_loss +
            self.w_obj_neg * obj_neg_loss +
            self.w_cls * cls_loss
        )

        # NaN/Inf safety check - replace with zero if detected
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
            box_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
            obj_pos_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
            obj_neg_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
            cls_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()

        loss_dict = {
            'box_loss': box_loss.float().item(),
            'obj_pos_loss': obj_pos_loss.float().item(),
            'obj_neg_loss': obj_neg_loss.float().item(),
            'cls_loss': cls_loss.float().item(),
            'total_loss': total_loss.float().item()
        }

        # Also return tensor losses for auto-weighting (with gradients)
        loss_tensors = {
            'box_loss': box_loss,
            'obj_pos_loss': obj_pos_loss,
            'obj_neg_loss': obj_neg_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict, total_loss, loss_tensors

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

    def _decode_boxes(self, offsets, anchors, stride):
        """
        Decode box predictions from offsets to xyxy coordinates.
        Uses shared YOLOBoxCodec to ensure consistency with inference.

        Args:
            offsets: [N, 4] raw network outputs (tx, ty, tw, th)
            anchors: [N, 4] anchor boxes (cx, cy, w, h)
            stride: Current feature map stride

        Returns:
            boxes: [N, 4] decoded boxes (x1, y1, x2, y2)
        """
        return YOLOBoxCodec.decode(offsets, anchors, stride)

    @staticmethod
    def _compute_iou(boxes1, boxes2):
        """
        Compute IoU between boxes.

        Args:
            boxes1: [N, 4] (x1, y1, x2, y2)
            boxes2: [M, 4] (cx, cy, w, h) - anchors in center format

        Returns:
            iou: [N, M]
        """
        # Convert boxes2 from (cx, cy, w, h) to (x1, y1, x2, y2)
        boxes2_xyxy = xywh_to_xyxy(boxes2)
        return box_iou(boxes1, boxes2_xyxy)


class AutomaticWeightedLoss(nn.Module):
    """
    Automatic loss weighting using Normalized Softplus.

    Why this instead of Uncertainty (log var)?
    The Uncertainty method allows weights to explode if one loss is numerically tiny
    (like obj_loss when averaged over all anchors), leading to negative total loss
    and ignoring other tasks.

    This implementation:
    1. Ensures weights are always positive (Softplus).
    2. Normalizes weights so they sum to N (number of losses).
    3. Prevents any single task from dominating or disappearing.
    """

    def __init__(self, num_losses: int = 4):
        """
        Args:
            num_losses: Number of loss terms to weight (default: 4 for box, obj_pos, obj_neg, cls)
        """
        super().__init__()
        # Initialize parameters to 0, softplus(0) ≈ 0.693
        # params[0]: box, params[1]: obj_pos, params[2]: obj_neg, params[3]: cls
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, box_loss: torch.Tensor, obj_pos_loss: torch.Tensor,
                obj_neg_loss: torch.Tensor, cls_loss: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute automatically weighted total loss.

        Args:
            box_loss: Box regression loss (CIoU)
            obj_pos_loss: Positive objectness loss (BCE)
            obj_neg_loss: Negative objectness loss (BCE)
            cls_loss: Classification loss (BCE)

        Returns:
            total_loss: Weighted sum of losses
            weight_dict: Dictionary with current weights for logging
        """
        losses = torch.stack([box_loss, obj_pos_loss, obj_neg_loss, cls_loss])

        # Compute weights: Softplus -> Normalize -> Scale to N
        # w_i = N * softplus(p_i) / sum(softplus(p_j))
        weights = F.softplus(self.params)
        weights = weights / (weights.sum() + 1e-6) * 4.0

        # Weighted sum
        total_loss = (weights * losses).sum()

        weight_dict = {
            'w_box': weights[0].item(),
            'w_obj_pos': weights[1].item(),
            'w_obj_neg': weights[2].item(),
            'w_cls': weights[3].item(),
        }

        return total_loss, weight_dict

    def get_weights(self) -> dict:
        """Get current effective weights without computing loss."""
        with torch.no_grad():
            weights = F.softplus(self.params)
            weights = weights / (weights.sum() + 1e-6) * 4.0
            return {
                'w_box': weights[0].item(),
                'w_obj_pos': weights[1].item(),
                'w_obj_neg': weights[2].item(),
                'w_cls': weights[3].item(),
            }
