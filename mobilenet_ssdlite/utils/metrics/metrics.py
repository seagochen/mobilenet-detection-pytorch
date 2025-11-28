"""
Evaluation metrics for object detection.
Computes mAP, Precision, Recall, and related metrics.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple

from ..ops.box_ops import box_iou_numpy as box_iou_batch


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个类别的AP

    Args:
        tp: (n_pred,) True positives
        conf: (n_pred,) Confidences
        pred_cls: (n_pred,) Predicted classes
        target_cls: (n_target,) Target classes
        eps: 小值防止除零

    Returns:
        p: Precision curve
        r: Recall curve
        ap: Average precision
        f1: F1 score
    """
    # 按置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到所有唯一的类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]

    # 创建结果数组
    px, py = np.linspace(0, 1, 1000), []  # Precision-recall curve points
    ap = np.zeros((nc, tp.shape[1]))  # AP for each IoU threshold
    p = np.zeros(nc)
    r = np.zeros(nc)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # 该类别的GT数量
        n_p = i.sum()  # 该类别的预测数量

        if n_p == 0 or n_l == 0:
            continue

        # 累积FP和TP
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # (n_pred, n_iou_thresholds)
        r[ci] = recall[-1, 0] if len(recall) > 0 else 0  # 最终recall值

        # Precision
        precision = tpc / (tpc + fpc)  # (n_pred, n_iou_thresholds)
        p[ci] = precision[-1, 0] if len(precision) > 0 else 0  # 最终precision值

        # AP (使用101点插值)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # F1 score
    f1 = 2 * p * r / (p + r + eps)

    return p, r, ap, f1


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算AP (VOC2010 11点插值方法)

    Args:
        recall: Recall curve
        precision: Precision curve

    Returns:
        ap: Average precision
        mpre: 插值precision
        mrec: 插值recall
    """
    # 添加哨兵值
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # 计算precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 计算PR曲线下面积 (使用101点插值)
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return ap, mpre, mrec


class DetectionMetrics:
    """
    检测指标计算器

    计算mAP@0.5, mAP@0.5:0.95等指标
    """

    def __init__(self, nc: int = 80):
        self.nc = nc
        self.stats = []  # List of (tp, conf, pred_cls, target_cls)

    def update(self, predictions: List[Dict], targets: List[np.ndarray], iou_thresholds: np.ndarray = None):
        """
        更新统计信息

        Args:
            predictions: List of predictions for each image
            targets: List of targets for each image
            iou_thresholds: IoU阈值数组
        """
        if iou_thresholds is None:
            iou_thresholds = np.array([0.5])  # 只计算 mAP@0.5，速度快10倍

        for pred, target in zip(predictions, targets):
            # 转换预测格式
            if len(pred) == 0:
                if len(target) > 0:
                    self.stats.append((
                        np.zeros((0, len(iou_thresholds)), dtype=bool),
                        np.array([]),
                        np.array([]),
                        target[:, 0]
                    ))
                continue

            # 安全地转换预测数据，处理可能的CUDA张量
            def to_numpy_safe(value):
                """安全地将值转换为numpy兼容格式"""
                if isinstance(value, torch.Tensor):
                    return value.cpu().item() if value.numel() == 1 else value.cpu().numpy()
                elif isinstance(value, (tuple, list)):
                    return tuple(to_numpy_safe(v) for v in value) if isinstance(value, tuple) else [to_numpy_safe(v) for v in value]
                return value

            pred_boxes = np.array([to_numpy_safe(p['bbox']) for p in pred])
            pred_conf = np.array([to_numpy_safe(p['confidence']) for p in pred])
            pred_cls = np.array([to_numpy_safe(p['class_id']) for p in pred])

            if len(target) == 0:
                self.stats.append((
                    np.zeros((len(pred), len(iou_thresholds)), dtype=bool),
                    pred_conf,
                    pred_cls,
                    np.array([])
                ))
                continue

            target_cls = target[:, 0]
            target_boxes = target[:, 1:]

            # 计算IoU
            iou = box_iou_batch(target_boxes, pred_boxes)

            # 匹配预测和GT（贪婪匹配，确保每个GT只被匹配一次）
            correct = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)

            # 按置信度从高到低排序预测框
            sort_idx = np.argsort(-pred_conf)

            for i, iou_thr in enumerate(iou_thresholds):
                detected_gt = set()  # 记录已被匹配的GT索引

                for pred_idx in sort_idx:
                    p_cls = pred_cls[pred_idx]

                    # 找到同类别且未被匹配的GT中IoU最大的
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, gt_cls in enumerate(target_cls):
                        if gt_idx in detected_gt:
                            continue
                        if p_cls != gt_cls:
                            continue
                        current_iou = iou[gt_idx, pred_idx]
                        if current_iou > iou_thr and current_iou > best_iou:
                            best_iou = current_iou
                            best_gt_idx = gt_idx

                    if best_gt_idx >= 0:
                        correct[pred_idx, i] = True
                        detected_gt.add(best_gt_idx)  # 标记该GT已被占用

            self.stats.append((correct, pred_conf, pred_cls, target_cls))

    def compute_metrics(self) -> Dict[str, float]:
        """
        计算最终指标

        Returns:
            metrics: 包含mAP@0.5等指标
        """
        if not self.stats:
            return {}

        # 合并所有统计
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        tp, conf, pred_cls, target_cls = stats

        # 计算AP
        p, r, ap, f1 = ap_per_class(tp, conf, pred_cls, target_cls)

        # 计算mAP@0.5（只有一个阈值时ap shape为(nc, 1)）
        ap50 = ap[:, 0] if ap.ndim > 1 else ap

        metrics = {
            'precision': p.mean(),
            'recall': r.mean(),
            'mAP@0.5': ap50.mean(),
            'f1': f1.mean()
        }

        return metrics

    def reset(self):
        """重置统计"""
        self.stats = []
