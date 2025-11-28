"""
Path resolution utilities for dataset loading.
Handles various dataset formats (Ultralytics, Roboflow, custom).
"""
from pathlib import Path


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
