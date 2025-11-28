"""
Anchor clustering utilities for object detection.
Uses K-Means clustering to compute optimal anchor sizes.
"""
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

from .path_utils import resolve_split_path, infer_label_dir
from ..ops.box_ops import box_iou_wh


def load_boxes_from_yolo_dataset(yaml_path: str, split: str = 'train', img_size: int = 640) -> np.ndarray:
    """
    从 YOLO 格式数据集加载所有边界框的宽高

    Args:
        yaml_path: 数据集 YAML 配置文件路径
        split: 数据集分割 ('train', 'val')
        img_size: 目标图像尺寸

    Returns:
        boxes_wh: (N, 2) 数组，包含所有框的 [width, height]（像素坐标）
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    root = Path(config.get('path', ''))
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    split_path = config.get(split)
    if split_path is None:
        raise ValueError(f"Split '{split}' not found in YAML config")

    img_dir = resolve_split_path(root, split_path)
    label_dir = infer_label_dir(img_dir)

    if not label_dir.exists():
        raise ValueError(f"Label directory does not exist: {label_dir}")

    # 收集所有框的宽高
    boxes_wh = []

    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                # YOLO 格式: class x_center y_center width height (normalized)
                w = float(parts[3]) * img_size
                h = float(parts[4]) * img_size

                if w > 0 and h > 0:
                    boxes_wh.append([w, h])

    return np.array(boxes_wh, dtype=np.float32)


def kmeans_anchors(
    boxes_wh: np.ndarray,
    n_anchors: int = 9,
    max_iter: int = 300,
    tol: float = 1e-5
) -> np.ndarray:
    """
    使用 K-Means 聚类计算最优 anchor 尺寸
    使用 1 - IoU 作为距离度量

    Args:
        boxes_wh: (N, 2) 框的宽高数组
        n_anchors: anchor 数量
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        anchors: (n_anchors, 2) 聚类中心 [width, height]
    """
    n_boxes = len(boxes_wh)
    if n_boxes < n_anchors:
        raise ValueError(f"Number of boxes ({n_boxes}) is less than n_anchors ({n_anchors})")

    # 随机初始化聚类中心
    np.random.seed(42)
    indices = np.random.choice(n_boxes, n_anchors, replace=False)
    centroids = boxes_wh[indices].copy()

    for iteration in range(max_iter):
        # 计算每个框到每个聚类中心的距离 (1 - IoU)
        distances = 1 - box_iou_wh(boxes_wh, centroids)

        # 分配每个框到最近的聚类中心
        assignments = np.argmin(distances, axis=1)

        # 更新聚类中心
        new_centroids = np.zeros_like(centroids)
        for i in range(n_anchors):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = boxes_wh[mask].mean(axis=0)
            else:
                # 如果没有框分配到这个聚类，随机选择一个框
                new_centroids[i] = boxes_wh[np.random.randint(n_boxes)]

        # 检查收敛
        diff = np.abs(new_centroids - centroids).sum()
        centroids = new_centroids

        if diff < tol:
            print(f"K-Means converged at iteration {iteration + 1}")
            break

    # 按面积排序
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    centroids = centroids[sorted_indices]

    return centroids


def compute_pos_neg_ratio(
    yaml_path: str,
    img_size: int = 640,
    strides: List[int] = [8, 16, 32],
    num_anchors_per_scale: int = 3
) -> float:
    """
    计算数据集的正负样本比例，用于确定 noobj_weight

    Args:
        yaml_path: 数据集 YAML 配置文件路径
        img_size: 目标图像尺寸
        strides: 各检测层的步长
        num_anchors_per_scale: 每个位置的 anchor 数量

    Returns:
        noobj_weight: 建议的负样本权重
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    root = Path(config.get('path', ''))
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    split_path = config.get('train')
    if split_path is None:
        raise ValueError("Split 'train' not found in YAML config")

    img_dir = resolve_split_path(root, split_path)
    label_dir = infer_label_dir(img_dir)

    if not label_dir.exists():
        raise ValueError(f"Label directory does not exist: {label_dir}")

    # 计算总 anchor 数量（每张图）
    total_anchors_per_image = sum(
        (img_size // s) ** 2 * num_anchors_per_scale
        for s in strides
    )

    # 统计所有图片的 GT 数量
    total_gt_boxes = 0
    num_images = 0

    for label_file in label_dir.glob('*.txt'):
        num_images += 1
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) >= 5:
                    total_gt_boxes += 1

    if num_images == 0:
        print("Warning: No images found, using default noobj_weight=1.0")
        return 1.0

    # 估算正样本数（假设每个 GT 匹配约 1-3 个 anchor）
    avg_gt_per_image = total_gt_boxes / num_images
    estimated_pos_per_image = avg_gt_per_image * 2  # 假设平均每个GT匹配2个anchor

    # 计算正负样本比例
    estimated_neg_per_image = total_anchors_per_image - estimated_pos_per_image
    pos_neg_ratio = estimated_pos_per_image / max(estimated_neg_per_image, 1)

    # 计算建议的 noobj_weight
    # 目标：让正负样本对 loss 的贡献大致相等
    # 如果正样本非常少，需要增加负样本的权重来平衡
    # noobj_weight = sqrt(pos_neg_ratio) 可以让两者贡献更接近
    # 限制在 [0.5, 2.0] 范围内
    noobj_weight = np.clip(np.sqrt(pos_neg_ratio) * 2, 0.5, 2.0)

    print(f"\nPos/Neg Sample Analysis:")
    print(f"  Total images: {num_images}")
    print(f"  Total GT boxes: {total_gt_boxes}")
    print(f"  Avg GT per image: {avg_gt_per_image:.1f}")
    print(f"  Total anchors per image: {total_anchors_per_image:,}")
    print(f"  Estimated pos/neg ratio: {pos_neg_ratio:.6f} ({pos_neg_ratio*100:.4f}%)")
    print(f"  Recommended noobj_weight: {noobj_weight:.3f}")

    return float(noobj_weight)


def compute_anchors_for_dataset(
    yaml_path: str,
    n_anchors: int = 9,
    img_size: int = 640,
    n_scales: int = 3
) -> Tuple[List[List[List[int]]], float]:
    """
    为数据集计算最优 anchor

    Args:
        yaml_path: 数据集 YAML 配置文件路径
        n_anchors: 总 anchor 数量（必须能被 n_scales 整除）
        img_size: 目标图像尺寸
        n_scales: 检测层数量

    Returns:
        anchors: 分配到各检测层的 anchor，格式为 [[[w,h], ...], ...]
    """
    assert n_anchors % n_scales == 0, f"n_anchors ({n_anchors}) must be divisible by n_scales ({n_scales})"

    print(f"Loading boxes from dataset: {yaml_path}")
    boxes_wh = load_boxes_from_yolo_dataset(yaml_path, 'train', img_size)
    print(f"Loaded {len(boxes_wh)} boxes")

    if len(boxes_wh) == 0:
        raise ValueError("No boxes found in dataset")

    print(f"Running K-Means clustering for {n_anchors} anchors...")
    anchors = kmeans_anchors(boxes_wh, n_anchors)

    # 分配到各检测层（按面积从小到大）
    anchors_per_scale = n_anchors // n_scales
    anchors_by_scale = []

    for i in range(n_scales):
        start_idx = i * anchors_per_scale
        end_idx = start_idx + anchors_per_scale
        scale_anchors = anchors[start_idx:end_idx]
        # 转换为整数列表
        anchors_by_scale.append([[int(round(w)), int(round(h))] for w, h in scale_anchors])

    return anchors_by_scale


def save_anchors(anchors: List[List[List[int]]], save_path: str, metadata: dict = None):
    """
    保存 anchor 配置到 YAML 文件

    Args:
        anchors: anchor 配置
        save_path: 保存路径
        metadata: 额外的元数据
    """
    data = {
        'anchors': anchors,
    }
    if metadata:
        data.update(metadata)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Anchors saved to: {save_path}")


def load_anchors(load_path: str) -> Optional[List[List[List[int]]]]:
    """
    从 YAML 文件加载 anchor 配置

    Args:
        load_path: 配置文件路径

    Returns:
        anchors: anchor 配置，如果文件不存在则返回 None
    """
    if not Path(load_path).exists():
        return None

    with open(load_path, 'r') as f:
        data = yaml.safe_load(f)

    return data.get('anchors')


def get_or_compute_anchors(
    save_dir: str,
    yaml_path: str,
    img_size: int = 640,
    n_anchors: int = 9,
    force_recompute: bool = False
) -> List[List[List[int]]]:
    """
    获取或计算 anchor 配置
    如果 save_dir 下存在 anchors.yaml 则加载，否则计算并保存

    Args:
        save_dir: 保存目录（通常是 runs/train/exp/）
        yaml_path: 数据集 YAML 配置文件路径
        img_size: 目标图像尺寸
        n_anchors: 总 anchor 数量
        force_recompute: 是否强制重新计算

    Returns:
        anchors: anchor 配置
    """
    anchor_file = Path(save_dir) / 'anchors.yaml'

    # 尝试加载已有配置
    if not force_recompute and anchor_file.exists():
        print(f"Loading existing anchors from: {anchor_file}")
        anchors = load_anchors(str(anchor_file))
        if anchors is not None:
            print(f"Loaded anchors: {anchors}")
            return anchors

    # 计算新的 anchor
    print("Computing optimal anchors for dataset...")
    anchors = compute_anchors_for_dataset(
        yaml_path=yaml_path,
        n_anchors=n_anchors,
        img_size=img_size
    )

    # 保存配置
    metadata = {
        'dataset': yaml_path,
        'img_size': img_size
    }
    save_anchors(anchors, str(anchor_file), metadata)

    print(f"Computed anchors: {anchors}")
    return anchors


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute optimal anchors for dataset')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--n-anchors', type=int, default=9, help='Number of anchors')
    parser.add_argument('--output', type=str, default='anchors.yaml', help='Output file')

    args = parser.parse_args()

    anchors = compute_anchors_for_dataset(
        args.data,
        args.n_anchors,
        args.img_size
    )

    print("\nComputed anchors (copy to your config):")
    print(f"anchors: {anchors}")

    save_anchors(anchors, args.output, {
        'dataset': args.data,
        'img_size': args.img_size
    })
