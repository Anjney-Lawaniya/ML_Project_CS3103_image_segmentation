"""Binary road segmentation metrics for model validation and evaluation."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute positive-class IoU, F1, precision, and recall for a batch."""
    predictions = torch.sigmoid(logits) >= threshold
    ground_truth = targets >= 0.5
    true_positives = int(torch.count_nonzero(predictions & ground_truth).item())
    false_positives = int(torch.count_nonzero(predictions & ~ground_truth).item())
    false_negatives = int(torch.count_nonzero(~predictions & ground_truth).item())

    iou_denominator = true_positives + false_positives + false_negatives
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    iou = true_positives / iou_denominator if iou_denominator else 0.0
    f1_score = 2 * true_positives / f1_denominator if f1_denominator else 0.0
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    return {
        "iou": float(iou),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
    }
