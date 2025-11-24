"""
MobileNet Backbone using timm library
Extracts multi-scale features for object detection
"""
import torch
import torch.nn as nn
import timm


class MobileNetBackbone(nn.Module):
    """
    MobileNet backbone that extracts features at multiple scales
    using timm pretrained models
    """

    def __init__(self, model_name='mobilenetv3_large_100', pretrained=True):
        """
        Args:
            model_name: timm model name (e.g., 'mobilenetv3_large_100', 'mobilenetv3_small_100')
            pretrained: whether to load pretrained weights
        """
        super().__init__()

        # Load MobileNet from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=(2, 3, 4)  # Extract features at 3 scales (stride 8, 16, 32)
        )

        # Get feature dimensions for each scale
        self.feature_info = self.backbone.feature_info.channels()

        print(f"Loaded {model_name} backbone")
        print(f"Feature channels at different scales: {self.feature_info}")

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            List of feature maps at different scales
            - features[0]: stride 8 (small objects)
            - features[1]: stride 16 (medium objects)
            - features[2]: stride 32 (large objects)
        """
        features = self.backbone(x)
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) for fusing multi-scale features
    """

    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of input channels from backbone [C3, C4, C5]
            out_channels: Output channels for all pyramid levels
        """
        super().__init__()

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions (3x3 conv to reduce aliasing)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: List of feature maps [P3, P4, P5] from backbone

        Returns:
            List of FPN feature maps at the same scales
        """
        # Lateral connections
        laterals = [
            lateral_conv(feature)
            for feature, lateral_conv in zip(features, self.lateral_convs)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level features
            upsampled = nn.functional.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
            # Add to current level
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply output convolutions
        outputs = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]

        return outputs


if __name__ == '__main__':
    # Test the backbone
    model = MobileNetBackbone('mobilenetv3_large_100', pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    features = model(x)

    print("\nBackbone output shapes:")
    for i, feat in enumerate(features):
        stride = 2 ** (i + 3)
        print(f"  Scale {i} (stride {stride}): {feat.shape}")

    # Test FPN
    fpn = FeaturePyramidNetwork(model.feature_info, out_channels=256)
    fpn_features = fpn(features)

    print("\nFPN output shapes:")
    for i, feat in enumerate(fpn_features):
        stride = 2 ** (i + 3)
        print(f"  Scale {i} (stride {stride}): {feat.shape}")
