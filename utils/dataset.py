"""
Dataset for object detection
Supports COCO-style annotations
"""
import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


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
