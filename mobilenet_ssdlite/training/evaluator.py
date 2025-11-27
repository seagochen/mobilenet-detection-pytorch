"""
Validation and evaluation utilities for object detection.
Provides model evaluation with mAP computation and visualization.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from ..utils import DetectionMetrics, plot_detection_samples


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    compute_metrics: bool = True,
    save_dir: Optional[Path] = None,
    plot_samples: int = 16
) -> Dict[str, float]:
    """
    Validate model on a dataset.

    Args:
        model: Detection model to evaluate
        dataloader: Validation dataloader
        criterion: Loss function for computing validation loss
        device: Device to run evaluation on
        class_names: List of class names
        compute_metrics: Whether to compute mAP and other metrics
        save_dir: Directory to save detection sample plots
        plot_samples: Number of detection samples to plot

    Returns:
        Dictionary containing validation metrics (val_loss, mAP@0.5, precision, recall, etc.)
    """
    model.eval()

    total_loss = 0
    loss_components = {}
    sample_images = []
    sample_preds = []
    sample_targets = []

    detection_metrics = DetectionMetrics(nc=len(class_names))

    pbar = tqdm(dataloader, desc='Validation')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        # Compute loss
        predictions_raw, anchors = model(images, targets)
        loss_dict, _ = criterion(predictions_raw, anchors, targets)

        total_loss += loss_dict['total_loss']
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

        if compute_metrics:
            # Inference prediction (with NMS)
            batch_dets = model.predict(images, conf_thresh=0.001, nms_thresh=0.6)

            # Format targets
            batch_targets_formatted = _format_targets(targets)

            # Format predictions
            batch_preds_formatted = _format_predictions(batch_dets)

            detection_metrics.update(batch_preds_formatted, batch_targets_formatted)

            # Collect plot samples
            if batch_idx == 0 and save_dir:
                n = min(plot_samples, len(images))
                for i in range(n):
                    sample_images.append(images[i].cpu().numpy())
                    sample_preds.append(batch_preds_formatted[i])
                    sample_targets.append(batch_targets_formatted[i])

    # Average losses
    num_batches = len(dataloader)
    metrics = {'val_loss': total_loss / num_batches}
    metrics.update({k: v / num_batches for k, v in loss_components.items()})

    # Compute mAP
    if compute_metrics:
        det_metrics = detection_metrics.compute_metrics()
        metrics.update(det_metrics)

        if save_dir and sample_images:
            plot_detection_samples(
                sample_images, sample_preds, sample_targets,
                class_names, save_dir, max_images=plot_samples
            )

    return metrics


def _format_targets(targets: List[Dict[str, torch.Tensor]]) -> List[np.ndarray]:
    """
    Format targets for metric computation.

    Args:
        targets: List of target dictionaries with 'boxes' and 'labels'

    Returns:
        List of numpy arrays with shape (n, 5) containing [label, x1, y1, x2, y2]
    """
    batch_targets_formatted = []
    for t in targets:
        boxes = t['boxes'].cpu().numpy()
        labels = t['labels'].cpu().numpy()
        if len(boxes) > 0:
            formatted = np.column_stack((labels, boxes))
            batch_targets_formatted.append(formatted)
        else:
            batch_targets_formatted.append(np.zeros((0, 5)))
    return batch_targets_formatted


def _format_predictions(batch_dets: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Format predictions for metric computation.

    Args:
        batch_dets: List of detection dictionaries with 'boxes', 'scores', 'labels'

    Returns:
        List of prediction lists, each containing dicts with 'bbox', 'confidence', 'class_id'
    """
    batch_preds_formatted = []
    for det in batch_dets:
        img_preds = []
        if len(det['boxes']) > 0:
            boxes = det['boxes'].cpu().numpy()
            scores = det['scores'].cpu().numpy()
            labels = det['labels'].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                img_preds.append({
                    'bbox': b,
                    'confidence': s,
                    'class_id': int(l)
                })
        batch_preds_formatted.append(img_preds)
    return batch_preds_formatted


class Evaluator:
    """
    Standalone evaluator for offline model evaluation.

    Can be used independently of the training pipeline for
    evaluating saved checkpoints on validation/test datasets.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        class_names: List[str],
        device: torch.device,
        conf_thresh: float = 0.001,
        nms_thresh: float = 0.6
    ):
        """
        Initialize evaluator.

        Args:
            model: Detection model to evaluate
            dataloader: Evaluation dataloader
            class_names: List of class names
            device: Device to run evaluation on
            conf_thresh: Confidence threshold for predictions
            nms_thresh: NMS IoU threshold
        """
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    @torch.no_grad()
    def evaluate(self, save_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Run evaluation on the dataset.

        Args:
            save_dir: Optional directory to save visualizations

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        detection_metrics = DetectionMetrics(nc=len(self.class_names))
        sample_images = []
        sample_preds = []
        sample_targets = []

        pbar = tqdm(self.dataloader, desc='Evaluating')

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)

            # Inference
            batch_dets = self.model.predict(
                images,
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh
            )

            # Format for metrics
            batch_targets_formatted = _format_targets(targets)
            batch_preds_formatted = _format_predictions(batch_dets)

            detection_metrics.update(batch_preds_formatted, batch_targets_formatted)

            # Collect samples for visualization
            if batch_idx == 0 and save_dir:
                n = min(16, len(images))
                for i in range(n):
                    sample_images.append(images[i].cpu().numpy())
                    sample_preds.append(batch_preds_formatted[i])
                    sample_targets.append(batch_targets_formatted[i])

        metrics = detection_metrics.compute_metrics()

        if save_dir and sample_images:
            plot_detection_samples(
                sample_images, sample_preds, sample_targets,
                self.class_names, save_dir, max_images=16
            )

        return metrics

    def print_results(self, metrics: Dict[str, float]) -> None:
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)

        main_metrics = ['mAP@0.5', 'precision', 'recall', 'f1']
        for key in main_metrics:
            if key in metrics:
                print(f"  {key:<15}: {metrics[key]:.4f}")

        print("-" * 50)
        print("Per-class AP:")
        for key, value in metrics.items():
            if key.startswith('AP_class_'):
                class_idx = int(key.split('_')[-1])
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
                print(f"  {class_name:<15}: {value:.4f}")

        print("=" * 50)
