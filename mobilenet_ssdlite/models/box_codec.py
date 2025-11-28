"""
Unified box encoding/decoding for YOLO-style detection.
Ensures consistency between training (loss computation) and inference (detection).
"""
import torch


class YOLOBoxCodec:
    """
    YOLO-style box encoder/decoder.
    Uses YOLOv5/v8 style encoding for numerical stability:
        - Center: (sigmoid(x) - 0.5) * 2 * stride + anchor_center
        - Size: (sigmoid(x) * 2) ** 2 * anchor_size
    """

    @staticmethod
    def decode(
        offsets: torch.Tensor,
        anchors: torch.Tensor,
        stride: int
    ) -> torch.Tensor:
        """
        Decode box predictions from network outputs to xyxy coordinates.

        Args:
            offsets: [..., 4] raw network outputs (tx, ty, tw, th)
            anchors: [..., 4] anchor boxes (cx, cy, w, h) in pixel coordinates
            stride: Feature map stride

        Returns:
            boxes: [..., 4] decoded boxes (x1, y1, x2, y2) in pixel coordinates
        """
        # Get anchor parameters
        anchor_cx = anchors[..., 0]
        anchor_cy = anchors[..., 1]
        anchor_w = anchors[..., 2]
        anchor_h = anchors[..., 3]

        # Decode center coordinates
        # (sigmoid(x) - 0.5) * 2 allows predictions in range [-1, 1] * stride
        pred_cx = (torch.sigmoid(offsets[..., 0]) - 0.5) * 2 * stride + anchor_cx
        pred_cy = (torch.sigmoid(offsets[..., 1]) - 0.5) * 2 * stride + anchor_cy

        # Decode width and height using YOLOv5/v8 style: (sigmoid(x)*2)^2
        # This is more stable than exp() and prevents gradient explosion
        # Output range: [0, 4] * anchor_size
        pred_w = (torch.sigmoid(offsets[..., 2]) * 2) ** 2 * anchor_w
        pred_h = (torch.sigmoid(offsets[..., 3]) * 2) ** 2 * anchor_h

        # Convert to xyxy format
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def decode_xywh(
        offsets: torch.Tensor,
        anchors: torch.Tensor,
        stride: int
    ) -> torch.Tensor:
        """
        Decode box predictions to center format (cx, cy, w, h).

        Args:
            offsets: [..., 4] raw network outputs (tx, ty, tw, th)
            anchors: [..., 4] anchor boxes (cx, cy, w, h) in pixel coordinates
            stride: Feature map stride

        Returns:
            boxes: [..., 4] decoded boxes (cx, cy, w, h) in pixel coordinates
        """
        anchor_cx = anchors[..., 0]
        anchor_cy = anchors[..., 1]
        anchor_w = anchors[..., 2]
        anchor_h = anchors[..., 3]

        pred_cx = (torch.sigmoid(offsets[..., 0]) - 0.5) * 2 * stride + anchor_cx
        pred_cy = (torch.sigmoid(offsets[..., 1]) - 0.5) * 2 * stride + anchor_cy
        pred_w = (torch.sigmoid(offsets[..., 2]) * 2) ** 2 * anchor_w
        pred_h = (torch.sigmoid(offsets[..., 3]) * 2) ** 2 * anchor_h

        return torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

    @staticmethod
    def encode(
        targets: torch.Tensor,
        anchors: torch.Tensor,
        stride: int,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Encode ground truth boxes to network target format.
        This is the inverse of decode().

        Args:
            targets: [..., 4] target boxes (x1, y1, x2, y2) in pixel coordinates
            anchors: [..., 4] anchor boxes (cx, cy, w, h) in pixel coordinates
            stride: Feature map stride
            eps: Small value for numerical stability

        Returns:
            offsets: [..., 4] encoded targets (tx, ty, tw, th)
        """
        # Convert targets to center format
        target_cx = (targets[..., 0] + targets[..., 2]) / 2
        target_cy = (targets[..., 1] + targets[..., 3]) / 2
        target_w = (targets[..., 2] - targets[..., 0]).clamp(min=eps)
        target_h = (targets[..., 3] - targets[..., 1]).clamp(min=eps)

        # Get anchor parameters
        anchor_cx = anchors[..., 0]
        anchor_cy = anchors[..., 1]
        anchor_w = anchors[..., 2].clamp(min=eps)
        anchor_h = anchors[..., 3].clamp(min=eps)

        # Encode center: inverse of (sigmoid(x) - 0.5) * 2 * stride + anchor_cx
        # target_cx = (sigmoid(tx) - 0.5) * 2 * stride + anchor_cx
        # (target_cx - anchor_cx) / (2 * stride) = sigmoid(tx) - 0.5
        # sigmoid(tx) = (target_cx - anchor_cx) / (2 * stride) + 0.5
        # tx = logit(sigmoid(tx))
        sigmoid_tx = (target_cx - anchor_cx) / (2 * stride) + 0.5
        sigmoid_ty = (target_cy - anchor_cy) / (2 * stride) + 0.5

        # Clamp to valid sigmoid range to avoid inf in logit
        sigmoid_tx = sigmoid_tx.clamp(min=eps, max=1 - eps)
        sigmoid_ty = sigmoid_ty.clamp(min=eps, max=1 - eps)

        tx = torch.logit(sigmoid_tx)
        ty = torch.logit(sigmoid_ty)

        # Encode size: inverse of (sigmoid(x) * 2) ** 2 * anchor_w
        # target_w = (sigmoid(tw) * 2) ** 2 * anchor_w
        # sqrt(target_w / anchor_w) = sigmoid(tw) * 2
        # sigmoid(tw) = sqrt(target_w / anchor_w) / 2
        sigmoid_tw = torch.sqrt(target_w / anchor_w) / 2
        sigmoid_th = torch.sqrt(target_h / anchor_h) / 2

        # Clamp to valid sigmoid range
        sigmoid_tw = sigmoid_tw.clamp(min=eps, max=1 - eps)
        sigmoid_th = sigmoid_th.clamp(min=eps, max=1 - eps)

        tw = torch.logit(sigmoid_tw)
        th = torch.logit(sigmoid_th)

        return torch.stack([tx, ty, tw, th], dim=-1)
