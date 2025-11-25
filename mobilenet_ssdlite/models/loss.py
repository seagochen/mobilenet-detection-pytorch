"""
YOLO Loss Functions for object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    """
    YOLO-style loss function
    Combines box regression, objectness, and classification losses
    """

    def __init__(self, num_classes, anchors, strides, input_size, loss_weights=None):
        """
        Args:
            num_classes: Number of object classes
            anchors: Anchor generator
            strides: Feature map strides [8, 16, 32]
            input_size: Input image size [H, W]
            loss_weights: Dict with 'box', 'obj', 'cls' weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.anchors = anchors
        self.strides = strides
        self.input_size = input_size

        # Loss weights
        if loss_weights is None:
            loss_weights = {'box': 0.05, 'obj': 1.0, 'cls': 0.5}
        self.box_weight = loss_weights['box']
        self.obj_weight = loss_weights['obj']
        self.cls_weight = loss_weights['cls']

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

        total_box_loss = 0
        total_obj_loss = 0
        total_cls_loss = 0
        total_num_pos = 0

        # Process each scale
        for scale_idx, (pred, anchor_boxes, stride) in enumerate(
            zip(predictions, anchors, self.strides)
        ):
            _, num_anchors, h, w, num_outputs = pred.shape

            # Split predictions
            box_pred = pred[..., :4]  # [B, num_anchors, H, W, 4]
            obj_pred = pred[..., 4]  # [B, num_anchors, H, W]
            cls_pred = pred[..., 5:]  # [B, num_anchors, H, W, num_classes]

            # Build targets for this scale
            # Use float32 for targets to avoid dtype mismatch with AMP
            box_target = torch.zeros(box_pred.shape, dtype=torch.float32, device=device)
            obj_target = torch.zeros(obj_pred.shape, dtype=torch.float32, device=device)
            cls_target = torch.zeros(cls_pred.shape, dtype=torch.float32, device=device)

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

                    # Compute box targets (offsets)
                    box_target[batch_idx][anchor_idx] = self._compute_box_targets(
                        assigned_targets, assigned_anchors
                    )

                    # Objectness target
                    obj_target[batch_idx][anchor_idx] = 1.0

                    # Class targets (one-hot)
                    cls_target[batch_idx][anchor_idx] = F.one_hot(
                        assigned_labels.long(),
                        num_classes=self.num_classes
                    ).float()

            # Compute losses
            # Box loss (only for positive samples)
            pos_mask = obj_target > 0
            num_pos = pos_mask.sum()

            if num_pos > 0:
                # IoU-based box loss (GIoU or CIoU can be used)
                box_loss = self._box_loss(
                    box_pred[pos_mask],
                    box_target[pos_mask],
                    anchor_boxes.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)[pos_mask]
                )
                total_box_loss += box_loss
                total_num_pos += num_pos

            # Objectness loss (all samples)
            obj_loss = self.bce_loss(obj_pred, obj_target).mean()
            total_obj_loss += obj_loss

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

        for i in range(len(gt_boxes)):
            gx, gy = gt_grid_x[i].item(), gt_grid_y[i].item()

            # Get anchors at this grid cell
            cell_anchors = anchors[:, gy, gx, :]  # [num_anchors, 4]

            # Compute IoU with all anchors at this location
            ious = self._compute_iou(
                gt_boxes[i:i+1],
                cell_anchors
            ).squeeze(0)  # [num_anchors]

            # Assign to anchor with highest IoU
            best_anchor = ious.argmax().item()
            best_iou = ious[best_anchor].item()

            key = (best_anchor, gy, gx)
            # Only update if this GT has higher IoU than existing assignment
            if key not in assignments or best_iou > assignments[key][0]:
                assignments[key] = (best_iou, i)

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

    def _compute_box_targets(self, gt_boxes, anchors):
        """
        Compute box regression targets (offsets from anchors)

        Args:
            gt_boxes: [N, 4] target boxes (x1, y1, x2, y2)
            anchors: [N, 4] anchor boxes (cx, cy, w, h)

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
        # tx, ty: normalized offset
        tx = (gt_cx - anchor_cx) / self.strides[0]  # Approximate
        ty = (gt_cy - anchor_cy) / self.strides[0]

        # tw, th: log-space scaling
        tw = torch.log(gt_w / anchor_w + 1e-16)
        th = torch.log(gt_h / anchor_h + 1e-16)

        targets = torch.stack([tx, ty, tw, th], dim=-1)
        return targets

    def _box_loss(self, pred, target, anchors):
        """
        Compute box regression loss (MSE on offsets)

        Args:
            pred: [N, 4] predicted offsets
            target: [N, 4] target offsets
            anchors: [N, 4] anchor boxes

        Returns:
            loss: Box regression loss
        """
        # Simple MSE loss on offsets
        loss = self.mse_loss(pred, target).sum(dim=-1).mean()
        return loss

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
