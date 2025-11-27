"""
Visualization utilities for object detection.

This module re-exports functions from plots.py for backward compatibility.
New code should import directly from plots.py.
"""

# Re-export all visualization functions from the unified module
from .plots import (
    visualize_detections,
    plot_training_curves_simple as plot_training_curves,
    get_color_palette,
    TrainingPlotter,
    plot_detection_samples,
    plot_labels_distribution,
)

__all__ = [
    'visualize_detections',
    'plot_training_curves',
    'get_color_palette',
    'TrainingPlotter',
    'plot_detection_samples',
    'plot_labels_distribution',
]
