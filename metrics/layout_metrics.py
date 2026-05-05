"""
Layout analysis metrics for text-line segmentation.
Implements mIoU and F1@IoU metrics from the paper.
"""

import numpy as np
from typing import List, Tuple


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_miou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> float:
    """
    Calculate mean IoU across all instances.
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
    
    Returns:
        Mean IoU
    """
    ious = []
    
    for pred, gt in zip(pred_masks, gt_masks):
        iou = calculate_iou(pred, gt)
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def calculate_f1_iou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Calculate F1 score at given IoU threshold.
    
    A prediction is considered correct if IoU >= threshold.
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
        iou_threshold: IoU threshold for matching
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return 1.0, 1.0, 1.0
    
    if len(pred_masks) == 0:
        return 0.0, 0.0, 0.0
    
    if len(gt_masks) == 0:
        return 0.0, 0.0, 0.0
    
    # Match predictions to ground truth
    matched_gt = set()
    true_positives = 0
    
    for pred in pred_masks:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_masks):
            if i in matched_gt:
                continue
            
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / len(pred_masks)
    recall = true_positives / len(gt_masks)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def calculate_mask_iou(
    pred_logits: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate IoU from model logits.
    
    Args:
        pred_logits: Prediction logits (H, W) or (N, H, W)
        gt_masks: Ground truth binary masks (H, W) or (N, H, W)
        threshold: Threshold for binarization
    
    Returns:
        Mean IoU
    """
    # Binarize predictions
    pred_masks = (pred_logits > threshold).astype(np.uint8)
    
    # Handle single mask case
    if pred_masks.ndim == 2:
        pred_masks = pred_masks[np.newaxis, ...]
        gt_masks = gt_masks[np.newaxis, ...]
    
    ious = []
    for pred, gt in zip(pred_masks, gt_masks):
        iou = calculate_iou(pred, gt)
        ious.append(iou)
    
    return np.mean(ious)


def compute_layout_metrics(predictions: List[dict], ground_truths: List[dict]) -> dict:
    """
    Compute all layout metrics for a batch.
    
    Args:
        predictions: List of prediction dicts with 'masks' key
        ground_truths: List of ground truth dicts with 'masks' key
    
    Returns:
        Dictionary with mIoU, F1@0.5, F1@0.75
    """
    all_ious = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_masks = pred['masks']
        gt_masks = gt['masks']
        
        # Calculate per-image IoU
        for p, g in zip(pred_masks, gt_masks):
            iou = calculate_iou(p, g)
            all_ious.append(iou)
    
    miou = np.mean(all_ious) if all_ious else 0.0
    
    # Calculate F1 at different thresholds
    _, _, f1_50 = calculate_f1_iou(
        [p['masks'] for p in predictions],
        [g['masks'] for g in ground_truths],
        iou_threshold=0.5
    )
    
    _, _, f1_75 = calculate_f1_iou(
        [p['masks'] for p in predictions],
        [g['masks'] for g in ground_truths],
        iou_threshold=0.75
    )
    
    return {
        'mIoU': miou,
        'F1@0.5': f1_50,
        'F1@0.75': f1_75,
    }


# Placeholder for detectron2/mmdet integration
def convert_to_detectron2_format(masks, boxes=None, labels=None):
    """Convert masks to detectron2 Instances format."""
    # This is a placeholder for integration with detectron2
    # In practice, you would return detectron2.structures.Instances
    pass


def convert_from_detectron2_format(instances):
    """Convert detectron2 Instances to standard format."""
    # This is a placeholder for integration with detectron2
    pass
