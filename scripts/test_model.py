"""
Test script to verify model architecture and forward pass
"""
import sys
sys.path.append('..')

import torch
import yaml

from models import MobileNetYOLO


def test_model():
    """Test model creation and forward pass"""

    # Load config
    with open('../configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Testing MobileNet-YOLO Model")
    print("=" * 80)

    # Create model
    print("\n1. Creating model...")
    model = MobileNetYOLO(config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # Test training mode
    print("\n2. Testing training mode...")
    model.train()
    batch_size = 2
    input_size = config['model']['input_size'][0]

    x = torch.randn(batch_size, 3, input_size, input_size)
    print(f"   Input shape: {x.shape}")

    predictions, anchors = model(x)

    print(f"   Number of prediction scales: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"   Scale {i}: {pred.shape} (B, num_anchors, H, W, outputs)")

    print(f"   Number of anchor scales: {len(anchors)}")
    for i, anchor in enumerate(anchors):
        print(f"   Anchors {i}: {anchor.shape} (num_anchors, H, W, 4)")

    # Test inference mode
    print("\n3. Testing inference mode...")
    model.eval()

    with torch.no_grad():
        detections = model.predict(x, conf_thresh=0.25, nms_thresh=0.45)

    print(f"   Number of images: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"   Image {i}: {det['boxes'].shape[0]} detections")
        print(f"      Boxes shape: {det['boxes'].shape}")
        print(f"      Scores shape: {det['scores'].shape}")
        print(f"      Labels shape: {det['labels'].shape}")

    # Test inference speed
    print("\n4. Testing inference speed...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    model = model.to(device)
    x = x.to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.predict(x, conf_thresh=0.25, nms_thresh=0.45)

    # Measure speed
    if device.type == 'cuda':
        torch.cuda.synchronize()

    import time
    num_iterations = 100
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model.predict(x, conf_thresh=0.25, nms_thresh=0.45)

        if device.type == 'cuda':
            torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    fps = batch_size / avg_time

    print(f"   Average inference time: {avg_time * 1000:.2f} ms")
    print(f"   Throughput: {fps:.2f} FPS")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()
