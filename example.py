"""
Simple example showing how to use MobileNet-YOLO
"""
import torch
import yaml
import cv2
import numpy as np

from models import MobileNetYOLO
from utils.visualize import visualize_detections


def main():
    # Load configuration
    print("Loading configuration...")
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    print("Creating model...")
    model = MobileNetYOLO(config)

    # Load checkpoint (if available)
    checkpoint_path = 'checkpoints/best.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print("No checkpoint found, using randomly initialized model")

    # Set to evaluation mode
    model.eval()

    # Create a dummy image for demonstration
    # In practice, load a real image with cv2.imread()
    print("\nCreating dummy image...")
    image = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)

    # Preprocess image
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        detections = model.predict(
            image_tensor,
            conf_thresh=0.25,  # Confidence threshold
            nms_thresh=0.45    # NMS IoU threshold
        )[0]

    # Print results
    num_detections = len(detections['boxes'])
    print(f"\nFound {num_detections} detections")

    if num_detections > 0:
        print("\nDetection details:")
        boxes = detections['boxes'].numpy()
        scores = detections['scores'].numpy()
        labels = detections['labels'].numpy()

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            class_name = config['data']['names'][int(label)]
            print(f"  {i+1}. {class_name}")
            print(f"     Confidence: {score:.3f}")
            print(f"     Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # Visualize results
        print("\nVisualizing detections...")
        vis_image = visualize_detections(
            image,
            detections,
            config['data']['names'],
            conf_thresh=0.25
        )

        # Save visualization
        output_path = 'example_output.jpg'
        cv2.imwrite(output_path, vis_image)
        print(f"Saved visualization to {output_path}")

    else:
        print("No objects detected (this is expected with a random image)")

    # Model information
    print("\n" + "=" * 60)
    print("Model Information:")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"Input size: {config['model']['input_size']}")
    print(f"Number of classes: {config['model']['num_classes']}")
    print(f"Backbone: {config['model']['backbone']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
