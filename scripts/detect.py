"""
Inference script for MobileNet-YOLO
Run object detection on images or videos
"""
import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm

from mobilenet_ssdlite.models import MobileNetYOLO
from mobilenet_ssdlite.utils.visualize import visualize_detections


def parse_args():
    parser = argparse.ArgumentParser(description='Run MobileNet-YOLO inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, folder, or video')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to txt files')
    return parser.parse_args()


def preprocess_image(image, input_size):
    """
    Preprocess image for model input

    Args:
        image: Input image (H, W, 3) numpy array
        input_size: Target input size

    Returns:
        image_tensor: Preprocessed image [1, 3, H, W]
        scale: Resize scale factor
        pad: Padding (pad_h, pad_w)
    """
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h))

    # Pad to square
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    padded = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # Convert to tensor
    image_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor, scale, (pad_h, pad_w)


def postprocess_detections(detections, scale, pad, original_shape):
    """
    Postprocess detections to original image coordinates

    Args:
        detections: Detection dict from model
        scale: Resize scale factor
        pad: Padding (pad_h, pad_w)
        original_shape: Original image shape (H, W)

    Returns:
        Postprocessed detections
    """
    boxes = detections['boxes'].clone()

    # Remove padding
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=original_shape[1] * scale)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=original_shape[0] * scale)

    # Scale back to original size
    boxes = boxes / scale

    detections['boxes'] = boxes
    return detections


def detect_image(model, image_path, config, args):
    """
    Run detection on a single image

    Args:
        model: MobileNet-YOLO model
        image_path: Path to image
        config: Configuration dict
        args: Command line arguments

    Returns:
        vis_image: Visualization image
        detections: Detection results
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]

    # Preprocess
    input_size = config['model']['input_size'][0]
    image_tensor, scale, pad = preprocess_image(image, input_size)
    image_tensor = image_tensor.to(args.device)

    # Run inference
    with torch.no_grad():
        detections = model.predict(
            image_tensor,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh
        )[0]

    # Postprocess
    detections = postprocess_detections(detections, scale, pad, original_shape)

    # Visualize
    class_names = config['data']['names']
    vis_image = visualize_detections(
        image,
        detections,
        class_names,
        conf_thresh=args.conf_thresh
    )

    return vis_image, detections


def detect_video(model, video_path, config, args, output_path):
    """
    Run detection on a video

    Args:
        model: MobileNet-YOLO model
        video_path: Path to video
        config: Configuration dict
        args: Command line arguments
        output_path: Output video path
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    # Process frames
    pbar = tqdm(total=total_frames, desc='Processing')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        # Preprocess
        input_size = config['model']['input_size'][0]
        image_tensor, scale, pad = preprocess_image(image, input_size)
        image_tensor = image_tensor.to(args.device)

        # Run inference
        with torch.no_grad():
            detections = model.predict(
                image_tensor,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh
            )[0]

        # Postprocess
        detections = postprocess_detections(detections, scale, pad, original_shape)

        # Visualize
        class_names = config['data']['names']
        vis_image = visualize_detections(
            image,
            detections,
            class_names,
            conf_thresh=args.conf_thresh
        )

        # Write frame
        out.write(vis_image)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    print(f"Saved output video to: {output_path}")


def save_detections_txt(detections, output_path, image_shape):
    """
    Save detections to txt file (YOLO format)

    Args:
        detections: Detection dict
        output_path: Output txt file path
        image_shape: Original image shape (H, W)
    """
    h, w = image_shape

    with open(output_path, 'w') as f:
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box

            # Convert to YOLO format (normalized center coordinates and size)
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.6f}\n")


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Loading model...")
    model = MobileNetYOLO(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Check source type
    source = args.source

    if os.path.isfile(source):
        # Single image or video
        ext = os.path.splitext(source)[1].lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Image
            print(f"Processing image: {source}")
            vis_image, detections = detect_image(model, source, config, args)

            # Save visualization
            output_path = os.path.join(
                args.output,
                os.path.basename(source)
            )
            cv2.imwrite(output_path, vis_image)
            print(f"Saved result to: {output_path}")

            # Save txt if requested
            if args.save_txt:
                txt_path = os.path.splitext(output_path)[0] + '.txt'
                image = cv2.imread(source)
                save_detections_txt(detections, txt_path, image.shape[:2])
                print(f"Saved detections to: {txt_path}")

            # Print detection results
            print(f"\nDetections: {len(detections['boxes'])}")
            for i, (box, score, label) in enumerate(zip(
                detections['boxes'].cpu().numpy(),
                detections['scores'].cpu().numpy(),
                detections['labels'].cpu().numpy()
            )):
                class_name = config['data']['names'][int(label)]
                print(f"  {i+1}. {class_name}: {score:.3f}")

        elif ext in ['.mp4', '.avi', '.mov']:
            # Video
            output_path = os.path.join(
                args.output,
                os.path.basename(source)
            )
            detect_video(model, source, config, args, output_path)

    elif os.path.isdir(source):
        # Directory of images
        print(f"Processing images in directory: {source}")

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        print(f"Found {len(image_files)} images")

        for image_file in tqdm(image_files, desc='Processing'):
            image_path = os.path.join(source, image_file)
            vis_image, detections = detect_image(model, image_path, config, args)

            # Save visualization
            output_path = os.path.join(args.output, image_file)
            cv2.imwrite(output_path, vis_image)

            # Save txt if requested
            if args.save_txt:
                txt_path = os.path.splitext(output_path)[0] + '.txt'
                image = cv2.imread(image_path)
                save_detections_txt(detections, txt_path, image.shape[:2])

        print(f"Saved results to: {args.output}")

    else:
        print(f"Error: Source not found: {source}")


if __name__ == '__main__':
    main()
