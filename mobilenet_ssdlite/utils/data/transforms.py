"""
Data augmentation transforms for object detection
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(config, is_train=True):
    """
    Get data augmentation transforms

    Args:
        config: Configuration dict
        is_train: Whether for training or validation

    Returns:
        Albumentations compose transform
    """
    if is_train:
        aug_config = config.get('train', {}).get('augmentation', {})

        transforms = A.Compose([
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)),
            A.VerticalFlip(p=aug_config.get('flipud', 0.0)),

            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),

            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.2),

            A.Affine(
                scale=(1 - aug_config.get('scale', 0.5), 1 + aug_config.get('scale', 0.5)),
                translate_percent={
                    'x': (-aug_config.get('translate', 0.1), aug_config.get('translate', 0.1)),
                    'y': (-aug_config.get('translate', 0.1), aug_config.get('translate', 0.1))
                },
                rotate=(-aug_config.get('degrees', 0.0), aug_config.get('degrees', 0.0)),
                shear=(-aug_config.get('shear', 0.0), aug_config.get('shear', 0.0)),
                p=0.5
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # (x1, y1, x2, y2)
            label_fields=['labels'],
            min_visibility=0.3
        ))
    else:
        # Validation: no augmentation
        transforms = A.Compose([
            A.NoOp()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))

    return transforms


if __name__ == '__main__':
    # Test transforms
    config = {
        'train': {
            'augmentation': {
                'fliplr': 0.5,
                'flipud': 0.0,
                'scale': 0.5,
                'translate': 0.1,
                'degrees': 10.0,
                'shear': 0.0
            }
        }
    }

    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)

    print("Train transforms:")
    print(train_transforms)
    print("\nVal transforms:")
    print(val_transforms)
