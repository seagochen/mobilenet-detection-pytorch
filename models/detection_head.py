"""
YOLO-style Detection Head for MobileNet backbone
"""
import torch
import torch.nn as nn
import numpy as np


class DetectionHead(nn.Module):
    """
    YOLO-style detection head that predicts bounding boxes,
    objectness scores, and class probabilities
    """

    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        """
        Args:
            in_channels: Input channels from FPN
            num_classes: Number of object classes
            num_anchors: Number of anchors per grid cell
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Output: (x, y, w, h, objectness, class_probs)
        # Total outputs per anchor: 5 + num_classes
        self.num_outputs = 5 + num_classes

        # Detection head: predicts box coordinates, objectness, and class scores
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * self.num_outputs, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            predictions: [B, num_anchors, H, W, num_outputs]
        """
        batch_size = x.size(0)
        h, w = x.shape[-2:]

        # Apply detection head
        out = self.conv(x)  # [B, num_anchors * num_outputs, H, W]

        # Reshape to [B, num_anchors, num_outputs, H, W]
        out = out.view(batch_size, self.num_anchors, self.num_outputs, h, w)

        # Permute to [B, num_anchors, H, W, num_outputs]
        out = out.permute(0, 1, 3, 4, 2).contiguous()

        return out


class AnchorGenerator:
    """
    Generate anchors for YOLO-style detection
    """

    def __init__(self, anchor_sizes, strides):
        """
        Args:
            anchor_sizes: List of anchor sizes for each scale
                         e.g., [[[10,13], [16,30], [33,23]], ...]
            strides: Feature map strides [8, 16, 32]
        """
        self.anchor_sizes = anchor_sizes
        self.strides = strides
        self.num_anchors = len(anchor_sizes[0])

    def generate_anchors(self, feature_shapes, device='cpu'):
        """
        Generate anchor boxes for all feature maps

        Args:
            feature_shapes: List of (H, W) for each feature map
            device: torch device

        Returns:
            anchors: List of anchor tensors for each scale
                    Each tensor shape: [num_anchors, H, W, 4] (cx, cy, w, h)
        """
        all_anchors = []

        for scale_idx, (h, w) in enumerate(feature_shapes):
            stride = self.strides[scale_idx]
            anchor_sizes = self.anchor_sizes[scale_idx]

            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )

            # Initialize anchors [num_anchors, H, W, 4]
            anchors = torch.zeros(
                self.num_anchors, h, w, 4,
                device=device, dtype=torch.float32
            )

            for anchor_idx, (aw, ah) in enumerate(anchor_sizes):
                # Center coordinates (in pixels at input resolution)
                anchors[anchor_idx, :, :, 0] = (grid_x + 0.5) * stride  # cx
                anchors[anchor_idx, :, :, 1] = (grid_y + 0.5) * stride  # cy

                # Width and height
                anchors[anchor_idx, :, :, 2] = aw  # w
                anchors[anchor_idx, :, :, 3] = ah  # h

            all_anchors.append(anchors)

        return all_anchors


class YOLODecoder:
    """
    Decode YOLO predictions into bounding boxes
    """

    def __init__(self, anchors, strides, input_size):
        """
        Args:
            anchors: Anchor generator
            strides: Feature map strides
            input_size: Input image size [H, W]
        """
        self.anchors = anchors
        self.strides = strides
        self.input_size = input_size

    def decode_predictions(self, predictions, anchor_boxes):
        """
        Decode YOLO predictions to absolute bounding boxes

        Args:
            predictions: List of prediction tensors for each scale
                        Each: [B, num_anchors, H, W, num_outputs]
            anchor_boxes: List of anchor tensors for each scale

        Returns:
            boxes: List of decoded boxes [B, num_anchors*H*W, 4] (x1, y1, x2, y2)
            scores: List of objectness scores [B, num_anchors*H*W]
            class_probs: List of class probabilities [B, num_anchors*H*W, num_classes]
        """
        all_boxes = []
        all_scores = []
        all_class_probs = []

        for pred, anchors, stride in zip(predictions, anchor_boxes, self.strides):
            batch_size, num_anchors, h, w, num_outputs = pred.shape

            # Split predictions
            box_pred = pred[..., :4]  # [B, num_anchors, H, W, 4]
            obj_pred = pred[..., 4:5]  # [B, num_anchors, H, W, 1]
            cls_pred = pred[..., 5:]  # [B, num_anchors, H, W, num_classes]

            # Decode box coordinates
            # tx, ty: offset from grid cell
            # tw, th: log-space offset from anchor
            anchors = anchors.unsqueeze(0)  # [1, num_anchors, H, W, 4]

            # Get anchor centers and sizes
            anchor_cx = anchors[..., 0]
            anchor_cy = anchors[..., 1]
            anchor_w = anchors[..., 2]
            anchor_h = anchors[..., 3]

            # Decode center coordinates
            pred_cx = (torch.sigmoid(box_pred[..., 0]) - 0.5) * 2 * stride + anchor_cx
            pred_cy = (torch.sigmoid(box_pred[..., 1]) - 0.5) * 2 * stride + anchor_cy

            # Decode width and height
            pred_w = torch.exp(box_pred[..., 2]) * anchor_w
            pred_h = torch.exp(box_pred[..., 3]) * anchor_h

            # Convert to (x1, y1, x2, y2)
            x1 = pred_cx - pred_w / 2
            y1 = pred_cy - pred_h / 2
            x2 = pred_cx + pred_w / 2
            y2 = pred_cy + pred_h / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            # Objectness scores
            scores = torch.sigmoid(obj_pred)

            # Class probabilities
            class_probs = torch.sigmoid(cls_pred)

            # Reshape for output
            boxes = boxes.view(batch_size, -1, 4)
            scores = scores.view(batch_size, -1)
            class_probs = class_probs.view(batch_size, -1, cls_pred.shape[-1])

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_class_probs.append(class_probs)

        # Concatenate all scales
        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        class_probs = torch.cat(all_class_probs, dim=1)

        return boxes, scores, class_probs


if __name__ == '__main__':
    # Test detection head
    head = DetectionHead(in_channels=256, num_classes=80, num_anchors=3)
    x = torch.randn(2, 256, 80, 80)
    out = head(x)
    print(f"Detection head output shape: {out.shape}")  # [2, 3, 80, 80, 85]

    # Test anchor generator
    anchor_sizes = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]]
    ]
    strides = [8, 16, 32]
    anchor_gen = AnchorGenerator(anchor_sizes, strides)

    feature_shapes = [(80, 80), (40, 40), (20, 20)]
    anchors = anchor_gen.generate_anchors(feature_shapes)

    print("\nGenerated anchors:")
    for i, anchor in enumerate(anchors):
        print(f"  Scale {i}: {anchor.shape}")
