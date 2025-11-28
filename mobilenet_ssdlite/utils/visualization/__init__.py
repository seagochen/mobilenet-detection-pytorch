"""
Visualization utilities for object detection.

This subpackage contains:
- Detection visualization
- Training plots (loss curves, metrics)
- Label distribution plots
"""

from .plots import (
    TrainingPlotter,
    plot_detection_samples,
    plot_labels_distribution,
    visualize_detections,
    get_color_palette,
)

__all__ = [
    'TrainingPlotter',
    'plot_detection_samples',
    'plot_labels_distribution',
    'visualize_detections',
    'get_color_palette',
]
