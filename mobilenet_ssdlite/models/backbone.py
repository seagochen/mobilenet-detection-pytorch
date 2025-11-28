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


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet) for multi-scale feature fusion.

    PANet adds a bottom-up path augmentation on top of FPN, allowing
    low-level features (with rich localization info) to flow to higher levels.

    Architecture:
        Backbone -> FPN (top-down) -> Bottom-up path -> Output

        C3 ──→ P3 ──────────────────→ N3 ──→ Output (stride 8)
               ↓ upsample              ↑ downsample
        C4 ──→ P4 ──────────────────→ N4 ──→ Output (stride 16)
               ↓ upsample              ↑ downsample
        C5 ──→ P5 ──────────────────→ N5 ──→ Output (stride 32)
    """

    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of input channels from backbone [C3, C4, C5]
            out_channels: Output channels for all pyramid levels
        """
        super().__init__()

        self.out_channels = out_channels

        # ===== Top-down pathway (FPN) =====
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])

        # ===== Bottom-up pathway (Path Aggregation) =====
        # Downsample convolutions (stride 2)
        self.downsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels_list) - 1)
        ])

        # Bottom-up output convolutions
        self.pan_convs = nn.ModuleList([
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
            features: List of feature maps [C3, C4, C5] from backbone

        Returns:
            List of PANet feature maps [N3, N4, N5]
        """
        # ===== Step 1: Top-down pathway (FPN) =====
        # Lateral connections
        laterals = [
            lateral_conv(feature)
            for feature, lateral_conv in zip(features, self.lateral_convs)
        ]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = nn.functional.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply FPN output convolutions -> P3, P4, P5
        fpn_outs = [
            fpn_conv(lateral)
            for lateral, fpn_conv in zip(laterals, self.fpn_convs)
        ]

        # ===== Step 2: Bottom-up pathway (Path Aggregation) =====
        # Start from P3 (lowest level, highest resolution)
        pan_outs = [fpn_outs[0]]  # N3 = P3 (no change for first level)

        for i in range(len(fpn_outs) - 1):
            # Downsample previous output
            downsampled = self.downsample_convs[i](pan_outs[-1])
            # Add to current FPN level
            fused = downsampled + fpn_outs[i + 1]
            # Apply output convolution
            pan_out = self.pan_convs[i + 1](fused)
            pan_outs.append(pan_out)

        # Apply convolution to first level as well
        pan_outs[0] = self.pan_convs[0](pan_outs[0])

        return pan_outs


def build_neck(neck_type, in_channels_list, out_channels=256):
    """
    Factory function to build neck module.

    Args:
        neck_type: 'fpn' or 'panet'
        in_channels_list: List of input channels from backbone
        out_channels: Output channels for all pyramid levels

    Returns:
        Neck module (FPN or PANet)
    """
    neck_type = neck_type.lower()
    if neck_type == 'fpn':
        return FeaturePyramidNetwork(in_channels_list, out_channels)
    elif neck_type == 'panet':
        return PANet(in_channels_list, out_channels)
    else:
        raise ValueError(f"Unknown neck type: {neck_type}. Choose from 'fpn' or 'panet'")


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
    print("\n" + "="*50)
    print("Testing FPN:")
    fpn = FeaturePyramidNetwork(model.feature_info, out_channels=256)
    fpn_features = fpn(features)

    print("FPN output shapes:")
    for i, feat in enumerate(fpn_features):
        stride = 2 ** (i + 3)
        print(f"  Scale {i} (stride {stride}): {feat.shape}")

    # Test PANet
    print("\n" + "="*50)
    print("Testing PANet:")
    panet = PANet(model.feature_info, out_channels=256)
    panet_features = panet(features)

    print("PANet output shapes:")
    for i, feat in enumerate(panet_features):
        stride = 2 ** (i + 3)
        print(f"  Scale {i} (stride {stride}): {feat.shape}")

    # Compare parameter counts
    print("\n" + "="*50)
    print("Parameter comparison:")
    fpn_params = sum(p.numel() for p in fpn.parameters())
    panet_params = sum(p.numel() for p in panet.parameters())
    print(f"  FPN params: {fpn_params:,}")
    print(f"  PANet params: {panet_params:,}")
    print(f"  PANet overhead: +{(panet_params - fpn_params):,} ({(panet_params/fpn_params - 1)*100:.1f}%)")
