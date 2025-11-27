"""
Path resolution utilities for dataset loading.
Handles various dataset formats (Ultralytics, Roboflow, custom).
"""
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple


def resolve_split_path(root: Path, split_path: str) -> Path:
    """
    Resolve split path to actual directory.
    Supports multiple Ultralytics/Roboflow dataset formats.

    Args:
        root: Dataset root directory
        split_path: Split path from YAML (e.g., 'train', 'images/train', '../train/images')

    Returns:
        Resolved absolute path to the split directory
    """
    root = Path(root).resolve()
    split_path_obj = Path(split_path)

    # Try multiple candidate paths
    candidates = [
        root / split_path_obj,
        (root / split_path_obj).resolve(),
    ]

    # Handle Roboflow '../' format
    path_str = str(split_path)
    if path_str.startswith('..'):
        clean_path = Path(path_str.lstrip('./').lstrip('..').lstrip('/'))
        candidates.append(root / clean_path)

    # Return first existing directory
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    # Fallback: return the basic resolution (may not exist)
    return (root / split_path_obj).resolve()


def infer_label_dir(img_dir: Path) -> Path:
    """
    Infer label directory from image directory.
    Common patterns:
        - images/train -> labels/train
        - train/images -> train/labels
        - train -> labels (parallel directory)

    Args:
        img_dir: Image directory path

    Returns:
        Inferred label directory path
    """
    img_dir = Path(img_dir)
    img_dir_str = str(img_dir)

    # Pattern 1: images -> labels replacement
    if 'images' in img_dir_str:
        label_dir = Path(img_dir_str.replace('images', 'labels'))
        if label_dir.exists():
            return label_dir

    # Pattern 2: Parallel 'labels' directory
    label_dir = img_dir.parent / 'labels'
    if label_dir.exists():
        return label_dir

    # Pattern 3: Same parent with 'labels' name
    label_dir = img_dir.parent / 'labels' / img_dir.name
    if label_dir.exists():
        return label_dir

    # Fallback: replace 'images' with 'labels' even if doesn't exist
    return Path(img_dir_str.replace('images', 'labels'))


def load_yolo_yaml(yaml_path: str) -> Dict:
    """
    Load YOLO dataset YAML configuration with resolved paths.

    Args:
        yaml_path: Path to dataset YAML file

    Returns:
        Config dict with resolved paths:
            - root: Resolved dataset root path
            - train_images: Resolved train images path
            - train_labels: Resolved train labels path
            - val_images: Resolved val images path (if exists)
            - val_labels: Resolved val labels path (if exists)
            - names: Class names list
            - nc: Number of classes
    """
    yaml_path = Path(yaml_path).resolve()

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve root path
    root = Path(config.get('path', ''))
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    result = {
        'root': root,
        'names': config.get('names', []),
        'nc': config.get('nc', len(config.get('names', []))),
        'yaml_path': yaml_path,
    }

    # Resolve split paths
    for split in ['train', 'val', 'test']:
        split_path = config.get(split)
        if split_path is not None:
            img_dir = resolve_split_path(root, split_path)
            label_dir = infer_label_dir(img_dir)
            result[f'{split}_images'] = img_dir
            result[f'{split}_labels'] = label_dir

    return result


def find_image_files(directory: Path, extensions: Tuple[str, ...] = None) -> list:
    """
    Find all image files in a directory.

    Args:
        directory: Directory to search
        extensions: Tuple of valid extensions (default: common image formats)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff')

    directory = Path(directory)
    if not directory.exists():
        return []

    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))

    return sorted(image_files)


def get_label_path(image_path: Path, label_dir: Path) -> Path:
    """
    Get corresponding label file path for an image.

    Args:
        image_path: Path to image file
        label_dir: Label directory

    Returns:
        Path to corresponding label file (.txt)
    """
    return label_dir / (image_path.stem + '.txt')


def validate_dataset_structure(config: Dict) -> Tuple[bool, str]:
    """
    Validate dataset directory structure.

    Args:
        config: Config dict from load_yolo_yaml()

    Returns:
        (is_valid, message) tuple
    """
    issues = []

    # Check train split
    if 'train_images' in config:
        if not config['train_images'].exists():
            issues.append(f"Train images not found: {config['train_images']}")
        if not config['train_labels'].exists():
            issues.append(f"Train labels not found: {config['train_labels']}")

    # Check val split
    if 'val_images' in config:
        if not config['val_images'].exists():
            issues.append(f"Val images not found: {config['val_images']}")
        if not config['val_labels'].exists():
            issues.append(f"Val labels not found: {config['val_labels']}")

    # Check class names
    if not config.get('names'):
        issues.append("No class names defined in YAML")

    if issues:
        return False, '\n'.join(issues)
    return True, "Dataset structure is valid"
