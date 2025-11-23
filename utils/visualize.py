"""
Visualization utilities for object detection
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_detections(image, detections, class_names, conf_thresh=0.5, save_path=None):
    """
    Visualize detection results on image

    Args:
        image: Input image [3, H, W] tensor or [H, W, 3] numpy array
        detections: Dict with 'boxes', 'scores', 'labels'
        class_names: List of class names
        conf_thresh: Confidence threshold for display
        save_path: Path to save visualization (optional)

    Returns:
        vis_image: Visualization image as numpy array
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()

    # Ensure image is in BGR for OpenCV
    if image.shape[2] == 3:
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        vis_image = image.copy()

    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']

    # Filter by confidence
    if isinstance(scores, torch.Tensor):
        mask = scores > conf_thresh
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask].cpu().numpy()
        labels = labels[mask].cpu().numpy()
    else:
        mask = scores > conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

    # Generate colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    # Draw boxes
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        label_idx = int(label)

        # Get color
        color = tuple(map(int, colors[label_idx % len(colors)]))

        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{class_names[label_idx]}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.rectangle(
            vis_image,
            (x1, y1 - text_h - 4),
            (x1 + text_w, y1),
            color,
            -1
        )

        cv2.putText(
            vis_image,
            label_text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, vis_image)

    return vis_image


def plot_training_curves(history, save_path=None):
    """
    Plot training curves

    Args:
        history: Dict with loss history
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    axes[0, 0].plot(history['total_loss'], label='Train')
    if 'val_total_loss' in history:
        axes[0, 0].plot(history['val_total_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Box loss
    axes[0, 1].plot(history['box_loss'], label='Train')
    axes[0, 1].set_title('Box Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Objectness loss
    axes[1, 0].plot(history['obj_loss'], label='Train')
    axes[1, 0].set_title('Objectness Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Class loss
    axes[1, 1].plot(history['cls_loss'], label='Train')
    axes[1, 1].set_title('Classification Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Test visualization
    image = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)

    detections = {
        'boxes': np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        'scores': np.array([0.9, 0.8]),
        'labels': np.array([0, 1])
    }

    class_names = ['person', 'car', 'dog']

    vis_image = visualize_detections(image, detections, class_names)
    print(f"Visualization shape: {vis_image.shape}")
