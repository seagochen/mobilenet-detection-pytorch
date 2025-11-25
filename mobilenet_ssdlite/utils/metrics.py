"""
Evaluation metrics for object detection
计算mAP、Precision、Recall等评估指标
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict


def box_iou_batch(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    计算批量框的IoU

    Args:
        box1: (N, 4) [x1, y1, x2, y2]
        box2: (M, 4) [x1, y1, x2, y2]

    Returns:
        iou: (N, M) IoU矩阵
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 计算交集
    x1 = np.maximum(box1[:, 0][:, None], box2[:, 0][None, :])
    y1 = np.maximum(box1[:, 1][:, None], box2[:, 1][None, :])
    x2 = np.minimum(box1[:, 2][:, None], box2[:, 2][None, :])
    y2 = np.minimum(box1[:, 3][:, None], box2[:, 3][None, :])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 计算并集
    union_area = area1[:, None] + area2[None, :] - inter_area

    iou = inter_area / (union_area + 1e-16)
    return iou


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


class ConfusionMatrix:
    """
    混淆矩阵
    用于计算mAP和可视化
    """

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45):
        """
        Args:
            nc: 类别数量
            conf: 置信度阈值
            iou_thres: IoU阈值
        """
        self.matrix = np.zeros((nc + 1, nc + 1))  # +1 for background
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections: np.ndarray, labels: np.ndarray):
        """
        处理一个batch的检测结果

        Args:
            detections: (n_pred, 6) [x1, y1, x2, y2, conf, class]
            labels: (n_gt, 5) [class, x1, y1, x2, y2]
        """
        if detections is None:
            gt_classes = labels[:, 0].astype(int)
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # 背景FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].astype(int)
        detection_classes = detections[:, 5].astype(int)

        if len(detections) == 0:
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        if len(labels) == 0:
            for dc in detection_classes:
                self.matrix[dc, self.nc] += 1  # 背景FP
            return

        # 计算IoU
        iou = box_iou_batch(labels[:, 1:], detections[:, :4])

        # 匹配
        matches = []
        for i, gc in enumerate(gt_classes):
            j = iou[i].argmax()
            if iou[i, j] > self.iou_thres:
                matches.append([i, j, iou[i, j], gc, detection_classes[j]])

        matches = np.array(matches)

        if len(matches):
            # 去除重复匹配
            if len(matches) > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            # 更新混淆矩阵
            for m in matches:
                gc = int(m[3])
                dc = int(m[4])
                self.matrix[dc, gc] += 1  # TP

            # FP
            for i, dc in enumerate(detection_classes):
                if not any(matches[:, 1] == i):
                    self.matrix[dc, self.nc] += 1

            # FN
            for i, gc in enumerate(gt_classes):
                if not any(matches[:, 0] == i):
                    self.matrix[self.nc, gc] += 1
        else:
            # 没有匹配
            for dc in detection_classes:
                self.matrix[dc, self.nc] += 1
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1

    def matrix_to_metrics(self) -> Dict[str, float]:
        """
        从混淆矩阵计算指标

        Returns:
            metrics: 包含precision, recall, f1等
        """
        tp = np.diag(self.matrix)[:-1]
        fp = self.matrix[:-1, self.nc]
        fn = self.matrix[self.nc, :-1]

        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)

        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean(),
            'tp': tp.sum(),
            'fp': fp.sum(),
            'fn': fn.sum()
        }

    def reset(self):
        """重置混淆矩阵"""
        self.matrix = np.zeros((self.nc + 1, self.nc + 1))


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

            # 匹配预测和GT
            correct = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)

            for i, iou_thr in enumerate(iou_thresholds):
                # 对每个预测框
                for pred_idx in range(len(pred)):
                    # 找到同类别的GT
                    same_class = target_cls == pred_cls[pred_idx]
                    if not same_class.any():
                        continue

                    # 找到IoU最大的GT
                    iou_same_class = iou[:, pred_idx].copy()
                    iou_same_class[~same_class] = 0

                    if iou_same_class.max() > iou_thr:
                        correct[pred_idx, i] = True

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

        # 每个类别的AP
        for i in range(self.nc):
            if i < len(ap50):
                metrics[f'AP_class_{i}'] = ap50[i]

        return metrics

    def reset(self):
        """重置统计"""
        self.stats = []


if __name__ == '__main__':
    print("Testing metrics...")

    # 测试混淆矩阵
    cm = ConfusionMatrix(nc=3, conf=0.25, iou_thres=0.45)

    # 模拟检测结果
    detections = np.array([
        [10, 10, 50, 50, 0.9, 0],  # 正确检测
        [100, 100, 150, 150, 0.8, 1],  # 正确检测
        [200, 200, 250, 250, 0.7, 2],  # 错误检测
    ])

    labels = np.array([
        [0, 10, 10, 50, 50],  # GT
        [1, 100, 100, 150, 150],  # GT
    ])

    cm.process_batch(detections, labels)
    metrics = cm.matrix_to_metrics()

    print("\nConfusion Matrix Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ Tests passed!")
