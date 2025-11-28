"""
Data transforms for object detection
Note: Data augmentation should be done externally (e.g., Roboflow)
"""
import albumentations as A


def get_transforms(config=None, is_train=True):
    """
    Get data transforms (no augmentation - handled by Roboflow)

    Args:
        config: Configuration dict (unused, kept for compatibility)
        is_train: Whether for training or validation (unused)

    Returns:
        Albumentations compose transform (no-op)
    """
    # No augmentation - data augmentation is handled externally by Roboflow
    transforms = A.Compose([
        A.NoOp()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # (x1, y1, x2, y2)
        label_fields=['labels']
    ))

    return transforms


if __name__ == '__main__':
    config = {}

    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    print("Train transforms:")
    print(train_transforms)
    print("\nVal transforms:")
    print(val_transforms)
