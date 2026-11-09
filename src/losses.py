"""Binary cross-entropy, Dice, and dynamic routing objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class BCEDiceLoss(nn.Module):
    """Average binary cross-entropy and smoothed soft Dice losses."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(dtype=logits.dtype)
        binary_cross_entropy_loss = self.binary_cross_entropy(logits, targets)
        dice_targets = targets.float()
        probabilities = torch.sigmoid(logits.float())
        intersection = torch.sum(probabilities * dice_targets, dim=(1, 2, 3))
        total_mass = torch.sum(probabilities + dice_targets, dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (total_mass + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return 0.5 * binary_cross_entropy_loss + 0.5 * dice_loss


class DynamicUNetLoss(nn.Module):
    """Combine primary, auxiliary deep supervision, and routing sparsity losses."""

    def __init__(
        self,
        auxiliary_weight: float = 0.2,
        flops_weight: float = 0.05,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.segmentation_loss = BCEDiceLoss(smooth=smooth)
        self.auxiliary_weight = auxiliary_weight
        self.flops_weight = flops_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        auxiliary_logits_list: Sequence[Tensor] | None = None,
        routing_masks_list: Sequence[Tensor] | None = None,
    ) -> Tensor:
        primary_loss = self.segmentation_loss(logits, targets)
        auxiliary_losses = []
        for auxiliary_logits in auxiliary_logits_list or ():
            resized_targets = functional.interpolate(
                targets.float(),
                size=auxiliary_logits.shape[-2:],
                mode="nearest",
            )
            auxiliary_losses.append(self.segmentation_loss(auxiliary_logits, resized_targets))
        if auxiliary_losses:
            auxiliary_loss = torch.stack(auxiliary_losses).mean()
        else:
            auxiliary_loss = primary_loss.new_zeros(())
        if routing_masks_list:
            flops_penalty = torch.stack(
                [routing_mask.to(dtype=logits.dtype).mean() for routing_mask in routing_masks_list]
            ).mean()
        else:
            flops_penalty = primary_loss.new_zeros(())
        return (
            primary_loss
            + self.auxiliary_weight * auxiliary_loss
            + self.flops_weight * flops_penalty
        )
