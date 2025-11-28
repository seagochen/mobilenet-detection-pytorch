"""
Data transforms for object detection.
Note: Data augmentation should be done externally (e.g., Roboflow).
"""
import albumentations as A


def get_transforms():
    """
    Get data transforms (no augmentation - handled by Roboflow).

    Returns:
        Albumentations compose transform (no-op)
    """
    return A.Compose([
        A.NoOp()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))
