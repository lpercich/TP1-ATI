import cv2
import numpy as np


def edge_density(edge_mask) -> float:
    edge_pixels = np.count_nonzero(edge_mask)
    total_pixels = edge_mask.size
    return edge_pixels / total_pixels


def connected_components_info(edge_mask):
    binary = (edge_mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    num_components = len(component_sizes)

    return num_components, component_sizes


def average_component_length(edge_mask) -> float:
    num_components, component_sizes = connected_components_info(edge_mask)

    if num_components == 0:
        return 0.0

    return float(np.mean(component_sizes))


def _dilate_binary_mask(binary_mask: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance <= 0:
        return binary_mask.copy()

    kernel_size = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(binary_mask, kernel)
    return (dilated > 0).astype(np.uint8)


def precision_recall_f1(pred_edges, gt_edges, tolerance: int = 1):
    pred = (pred_edges > 0).astype(np.uint8)
    gt = (gt_edges > 0).astype(np.uint8)

    pred_dil = _dilate_binary_mask(pred, tolerance)
    gt_dil = _dilate_binary_mask(gt, tolerance)

    true_positive_for_precision = np.sum((pred == 1) & (gt_dil == 1))
    true_positive_for_recall = np.sum((gt == 1) & (pred_dil == 1))

    pred_count = int(np.sum(pred))
    gt_count = int(np.sum(gt))

    precision = (
        true_positive_for_precision / pred_count if pred_count > 0 else 0.0
    )
    recall = (
        true_positive_for_recall / gt_count if gt_count > 0 else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)