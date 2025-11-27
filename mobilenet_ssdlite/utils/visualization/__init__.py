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
    plot_training_curves_simple as plot_training_curves,
)

__all__ = [
    'TrainingPlotter',
    'plot_detection_samples',
    'plot_labels_distribution',
    'visualize_detections',
    'get_color_palette',
    'plot_training_curves',
]
