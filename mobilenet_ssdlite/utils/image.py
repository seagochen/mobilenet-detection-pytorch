"""
Image normalization and denormalization utilities.
Provides consistent image preprocessing across training, inference, and visualization.
"""
import numpy as np
import torch

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalize_params(format='torch'):
    """
    Get normalization parameters in the specified format.

    Args:
        format: 'torch' for PyTorch tensor, 'numpy' for numpy array, 'tuple' for raw values

    Returns:
        mean, std in the requested format
    """
    if format == 'torch':
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    elif format == 'numpy':
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return mean, std


def normalize(img, inplace=False):
    """
    Normalize image using ImageNet mean and std.

    Args:
        img: Image tensor [C, H, W] or numpy array [H, W, C], values in [0, 1]
        inplace: If True, modify the input tensor in place

    Returns:
        Normalized image
    """
    if isinstance(img, torch.Tensor):
        mean, std = get_normalize_params('torch')
        mean = mean.to(img.device, img.dtype)
        std = std.to(img.device, img.dtype)
        if inplace:
            img.sub_(mean).div_(std)
            return img
        return (img - mean) / std
    else:
        mean, std = get_normalize_params('numpy')
        # Assume HWC format for numpy
        return (img - mean) / std


def denormalize(img, inplace=False):
    """
    Denormalize image from ImageNet normalization back to [0, 1] range.

    Args:
        img: Normalized image tensor [C, H, W] or numpy array [H, W, C]
        inplace: If True, modify the input tensor in place

    Returns:
        Denormalized image with values in [0, 1]
    """
    if isinstance(img, torch.Tensor):
        mean, std = get_normalize_params('torch')
        mean = mean.to(img.device, img.dtype)
        std = std.to(img.device, img.dtype)
        if inplace:
            img.mul_(std).add_(mean)
            return img.clamp_(0, 1)
        result = img * std + mean
        return result.clamp(0, 1)
    else:
        mean, std = get_normalize_params('numpy')
        # Assume HWC format for numpy
        result = img * std + mean
        return np.clip(result, 0, 1)


def denormalize_to_uint8(img):
    """
    Denormalize image and convert to uint8 [0, 255].

    Args:
        img: Normalized image tensor [C, H, W] or numpy array [H, W, C]

    Returns:
        Denormalized uint8 image
    """
    result = denormalize(img)
    if isinstance(result, torch.Tensor):
        return (result * 255).to(torch.uint8)
    else:
        return (result * 255).astype(np.uint8)
