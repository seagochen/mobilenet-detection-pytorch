"""
Training module for MobileNet-YOLO object detection.

This module provides:
- Training configuration and backbone registry
- Training pipeline (Trainer class)
- Validation and evaluation utilities
"""

from .config import (
    AVAILABLE_BACKBONES,
    DEFAULT_ANCHORS,
    list_available_backbones,
    validate_backbone,
    build_config,
    get_backbone_info,
    get_all_backbone_names,
)

from .trainer import Trainer

from .evaluator import (
    validate,
    Evaluator,
)

__all__ = [
    # Configuration
    'AVAILABLE_BACKBONES',
    'DEFAULT_ANCHORS',
    'list_available_backbones',
    'validate_backbone',
    'build_config',
    'get_backbone_info',
    'get_all_backbone_names',
    # Training
    'Trainer',
    # Evaluation
    'validate',
    'Evaluator',
]
