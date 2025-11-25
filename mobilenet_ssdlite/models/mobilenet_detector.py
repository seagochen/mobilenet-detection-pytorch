"""
MobileNet-YOLO: Object Detection Model
Combines MobileNet backbone with YOLO-style detection head
"""
import torch
import torch.nn as nn
import torchvision

from .backbone import MobileNetBackbone, FeaturePyramidNetwork
from .detection_head import DetectionHead, AnchorGenerator, YOLODecoder


class MobileNetDetector(nn.Module):
    """
    Complete MobileNet-YOLO object detection model
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()

        self.config = config
        self.num_classes = config['model']['num_classes']
        self.input_size = config['model']['input_size']
        self.num_anchors = config['model']['num_anchors']

        # Backbone: MobileNet from timm
        self.backbone = MobileNetBackbone(
            model_name=config['model']['backbone'],
            pretrained=config['model']['pretrained']
        )

        # Get backbone output channels
        backbone_channels = self.backbone.feature_info

        # Feature Pyramid Network
        fpn_channels = config['model']['fpn_channels']
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=backbone_channels,
            out_channels=fpn_channels
        )

        # Detection heads for each scale
        self.detection_heads = nn.ModuleList([
            DetectionHead(
                in_channels=fpn_channels,
                num_classes=self.num_classes,
                num_anchors=self.num_anchors
            )
            for _ in range(3)  # 3 scales
        ])

        # Anchor generator
        anchor_sizes = config['anchors']
        strides = [8, 16, 32]
        self.anchor_generator = AnchorGenerator(anchor_sizes, strides)
        self.strides = strides

        # YOLO decoder for inference
        self.decoder = YOLODecoder(
            anchors=self.anchor_generator,
            strides=strides,
            input_size=self.input_size
        )

        # Cache for anchor boxes
        self._anchor_cache = {}

    def forward(self, x, targets=None):
        """
        Forward pass

        Args:
            x: Input images [B, 3, H, W]
            targets: Ground truth boxes (for training) [optional]

        Returns:
            If training (targets provided):
                predictions: List of raw predictions for each scale
                anchors: List of anchor boxes for each scale
            If inference (no targets):
                boxes: Decoded bounding boxes
                scores: Objectness scores
                class_probs: Class probabilities
        """
        batch_size = x.size(0)
        device = x.device

        # Extract features from backbone
        features = self.backbone(x)

        # Apply FPN
        fpn_features = self.fpn(features)

        # Apply detection heads
        predictions = []
        for i, (feature, head) in enumerate(zip(fpn_features, self.detection_heads)):
            pred = head(feature)
            predictions.append(pred)

        # Generate anchors
        feature_shapes = [pred.shape[2:4] for pred in predictions]
        cache_key = str(feature_shapes)

        if cache_key not in self._anchor_cache or self._anchor_cache[cache_key][0].device != device:
            anchors = self.anchor_generator.generate_anchors(feature_shapes, device=device)
            self._anchor_cache[cache_key] = anchors
        else:
            anchors = self._anchor_cache[cache_key]

        # Training mode: return raw predictions and anchors
        if self.training or targets is not None:
            return predictions, anchors

        # Inference mode: decode predictions
        boxes, scores, class_probs = self.decoder.decode_predictions(predictions, anchors)

        return boxes, scores, class_probs

    def predict(self, x, conf_thresh=0.25, nms_thresh=0.45):
        """
        Make predictions with NMS post-processing (Optimized with torchvision.ops.nms)

        Args:
            x: Input images [B, 3, H, W]
            conf_thresh: Confidence threshold
            nms_thresh: NMS IoU threshold

        Returns:
            detections: List of detections for each image
                       Each detection: dict with 'boxes', 'scores', 'labels'
        """
        self.eval()

        with torch.no_grad():
            # 1. 获取模型输出 (boxes: [B, N, 4], scores: [B, N], probs: [B, N, C])
            boxes, obj_scores, class_probs = self.forward(x)

            # 2. 计算最终分数: objectness * class_prob
            # [B, N, num_classes]
            scores = obj_scores.unsqueeze(-1) * class_probs

            # 获取每个 anchor 最高分的类别
            # max_scores: [B, N], labels: [B, N]
            max_scores, labels = scores.max(dim=-1)

            batch_detections = []

            for i in range(boxes.size(0)):
                # 3. 阈值过滤
                mask = max_scores[i] > conf_thresh

                if not mask.any():
                    batch_detections.append({
                        'boxes': torch.tensor([], device=boxes.device),
                        'scores': torch.tensor([], device=boxes.device),
                        'labels': torch.tensor([], device=boxes.device)
                    })
                    continue

                img_boxes = boxes[i][mask]      # [M, 4]
                img_scores = max_scores[i][mask]  # [M]
                img_labels = labels[i][mask]     # [M]

                # 4. 类别隔离 NMS (Class-agnostic NMS with offsets)
                # 为避免不同类别的框相互抑制，给坐标加一个基于类别的偏移量
                max_coordinate = img_boxes.max() + 5000
                offsets = img_labels.float() * max_coordinate
                boxes_for_nms = img_boxes + offsets[:, None]

                # 5. 使用 torchvision 极速 NMS
                keep = torchvision.ops.nms(boxes_for_nms, img_scores, nms_thresh)

                batch_detections.append({
                    'boxes': img_boxes[keep],
                    'scores': img_scores[keep],
                    'labels': img_labels[keep]
                })

            return batch_detections


if __name__ == '__main__':
    # Test the complete model
    config = {
        'model': {
            'backbone': 'mobilenetv3_large_100',
            'pretrained': False,
            'num_classes': 80,
            'input_size': [640, 640],
            'fpn_channels': 256,
            'num_anchors': 3
        },
        'anchors': [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
    }

    model = MobileNetDetector(config)
    x = torch.randn(2, 3, 640, 640)

    # Test training mode
    model.train()
    predictions, anchors = model(x)
    print("Training mode:")
    print(f"  Number of scales: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  Scale {i}: {pred.shape}")

    # Test inference mode
    model.eval()
    detections = model.predict(x, conf_thresh=0.25, nms_thresh=0.45)
    print("\nInference mode:")
    for i, det in enumerate(detections):
        print(f"  Image {i}: {det['boxes'].shape[0]} detections")
