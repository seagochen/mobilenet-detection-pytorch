"""
Dataset for object detection
Supports COCO-style annotations and YOLO format
"""
import os
import json
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class DetectionDataset(Dataset):
    """
    Object detection dataset
    Expects data in COCO format or simple format
    """

    def __init__(self, data_dir, annotation_file=None, transforms=None, img_size=640):
        """
        Args:
            data_dir: Directory containing images
            annotation_file: Path to COCO-style JSON annotation file (optional)
            transforms: Albumentations transforms
            img_size: Target image size
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.img_size = img_size

        # Load annotations
        self.annotations = []
        self.images = []

        if annotation_file and os.path.exists(annotation_file):
            self._load_coco_annotations(annotation_file)
        else:
            # Simple mode: load all images in directory
            self._load_simple_mode()

    def _load_coco_annotations(self, annotation_file):
        """Load COCO-style annotations"""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Build image id to annotations mapping
        img_to_anns = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Process images
        for img_info in coco_data.get('images', []):
            img_id = img_info['id']
            img_path = os.path.join(self.data_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                continue

            # Get annotations for this image
            anns = img_to_anns.get(img_id, [])

            boxes = []
            labels = []

            for ann in anns:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

            self.images.append(img_path)
            self.annotations.append({
                'boxes': np.array(boxes, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64)
            })

    def _load_simple_mode(self):
        """Load images without annotations (for inference)"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        for filename in os.listdir(self.data_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                img_path = os.path.join(self.data_dir, filename)
                self.images.append(img_path)
                self.annotations.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros((0,), dtype=np.int64)
                })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image: [3, H, W] tensor
            target: dict with 'boxes' and 'labels'
        """
        img_path = self.images[idx]
        annotation = self.annotations[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = annotation['boxes'].copy()
        labels = annotation['labels'].copy()

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

        # Resize to target size
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h))

        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Scale boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            'boxes': torch.from_numpy(boxes).float() if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.from_numpy(labels).long() if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)
        }

        return image, target


class YOLODataset(Dataset):
    """
    YOLO format dataset
    Expects:
    - YAML config file with dataset configuration
    - Images in specified directories
    - Label files in YOLO format (txt files with normalized coordinates)
    """

    def __init__(self, yaml_path, split='train', transforms=None, img_size=640):
        """
        Args:
            yaml_path: Path to YOLO dataset YAML file
            split: Dataset split ('train', 'val', or 'test')
            transforms: Albumentations transforms
            img_size: Target image size
        """
        self.transforms = transforms
        self.img_size = img_size
        self.split = split

        # Load YAML config
        yaml_path = Path(yaml_path).resolve()
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get dataset root path
        # Support both 'path' field and relative paths from yaml location
        self.root = Path(self.config.get('path', ''))
        if not self.root.is_absolute():
            # If path is relative or empty, make it relative to the yaml file
            self.root = yaml_path.parent / self.root
        self.root = self.root.resolve()

        # Get class names and count
        self.class_names = self.config['names']
        self.num_classes = self.config.get('nc', len(self.class_names))

        # Get image directory for the split
        split_path = self.config.get(split)
        if split_path is None:
            raise ValueError(f"Split '{split}' not found in YAML config")

        # Resolve the image directory path
        # Handle both absolute and relative paths (including ../train/images style)
        self.img_dir = self._resolve_split_path(split_path)

        # Infer label directory (typically same structure but 'images' -> 'labels')
        self.label_dir = self._infer_label_dir(self.img_dir)

        # Load image paths
        self.images = []
        self._load_image_paths()

        print(f"Loaded {len(self.images)} images from {self.img_dir}")

    def _resolve_split_path(self, split_path):
        """
        Resolve split path to actual directory.
        Supports multiple Ultralytics/Roboflow dataset formats:
        - Absolute paths
        - Relative paths from root (e.g., 'train/images')
        - Relative paths from yaml (e.g., '../train/images')
        """
        split_path = Path(split_path)

        # Try different resolution strategies
        candidates = [
            # 1. Relative to root path
            self.root / split_path,
            # 2. Relative to yaml file (handles '../train/images' style)
            (self.root / split_path).resolve(),
        ]

        # 3. If path has '..' prefix, also try stripping it for Roboflow format
        # e.g., '../train/images' -> 'train/images' relative to yaml dir
        path_str = str(split_path)
        if path_str.startswith('..'):
            # Remove leading '../' and resolve from yaml parent
            clean_path = Path(path_str.lstrip('./').lstrip('..').lstrip('/'))
            candidates.append(self.root / clean_path)

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved

        # If none found, return the default (will raise error later)
        return (self.root / split_path).resolve()

    def _infer_label_dir(self, img_dir):
        """
        Infer label directory from image directory.
        Supports both 'images/train' -> 'labels/train' and 'train/images' -> 'train/labels'
        """
        img_dir_str = str(img_dir)

        if 'images' in img_dir_str:
            label_dir = Path(img_dir_str.replace('images', 'labels'))
            if label_dir.exists():
                return label_dir

        # Fallback: parallel directory structure
        return img_dir.parent / 'labels'

    def _load_image_paths(self):
        """Load all image paths from the image directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if not self.img_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.img_dir}")

        for ext in valid_extensions:
            self.images.extend(list(self.img_dir.glob(f'*{ext}')))
            self.images.extend(list(self.img_dir.glob(f'*{ext.upper()}')))

        self.images = sorted(self.images)

    def _load_yolo_labels(self, label_path):
        """
        Load YOLO format labels from txt file
        Format: class_id x_center y_center width height (normalized 0-1)

        Returns:
            boxes: numpy array of shape [N, 4] in xyxy format (pixel coordinates)
            labels: numpy array of shape [N] with class indices
        """
        boxes = []
        labels = []

        if not label_path.exists():
            # No labels for this image
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert from normalized xywh to normalized xyxy
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image: [3, H, W] tensor
            target: dict with 'boxes' and 'labels'
        """
        img_path = self.images[idx]

        # Get corresponding label file
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Load labels (normalized coordinates)
        boxes_norm, labels = self._load_yolo_labels(label_path)

        # Convert normalized coordinates to pixel coordinates
        boxes = boxes_norm.copy()
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

        # Resize to target size
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h))

        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Scale boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            'boxes': torch.from_numpy(boxes).float() if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.from_numpy(labels).long() if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)
        }

        return image, target


def collate_fn(batch):
    """
    Custom collate function for batching detection samples

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: [B, 3, H, W] tensor
        targets: List of target dicts
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


if __name__ == '__main__':
    # Test dataset
    dataset = DetectionDataset(
        data_dir='data/train',
        annotation_file=None,
        transforms=None,
        img_size=640
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels shape: {target['labels'].shape}")
