"""Micro segmentation metrics derived from binary confusion counts."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_confusion_counts(
    logits: Tensor, targets: Tensor, threshold: float = 0.5
) -> dict[str, int]:
    """Compute pixel-level TP, FP, FN, and TN counts for a binary mask."""
    predictions = torch.sigmoid(logits) >= threshold
    ground_truth = targets >= 0.5
    return {
        "true_positives": int((predictions & ground_truth).sum().item()),
        "false_positives": int((predictions & ~ground_truth).sum().item()),
        "false_negatives": int((~predictions & ground_truth).sum().item()),
        "true_negatives": int((~predictions & ~ground_truth).sum().item()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute positive-class IoU, F1, precision, and recall from counts."""
    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    if true_positives == 0 and false_positives == 0 and false_negatives == 0:
        return {"iou": 1.0, "f1_score": 1.0, "precision": 1.0, "recall": 1.0}
    iou_denominator = true_positives + false_positives + false_negatives
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return {
        "iou": true_positives / iou_denominator if iou_denominator else 0.0,
        "f1_score": 2 * true_positives / f1_denominator if f1_denominator else 0.0,
        "precision": true_positives / precision_denominator if precision_denominator else 0.0,
        "recall": true_positives / recall_denominator if recall_denominator else 0.0,
    }


def compute_segmentation_metrics(
    logits: Tensor, targets: Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Compute pixel-level IoU, F1, precision, and recall for a binary mask."""
    return metrics_from_counts(compute_confusion_counts(logits, targets, threshold))
