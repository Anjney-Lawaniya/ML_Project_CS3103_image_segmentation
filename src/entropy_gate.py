"""Parameter-free Shannon entropy routing with a straight-through estimator."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StraightThroughEstimator(torch.autograd.Function):
    """Binarize continuous entropy while passing upstream gradients unchanged."""

    @staticmethod
    def forward(ctx: object, entropy: Tensor, threshold: float) -> Tensor:
        return (entropy > threshold).to(dtype=entropy.dtype)

    @staticmethod
    def backward(ctx: object, output_gradient: Tensor) -> tuple[Tensor, None]:
        return output_gradient, None


class ShannonEntropyGate(nn.Module):
    """Threshold normalized binary entropy from one-channel auxiliary logits."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.eps = float(eps)

    def forward(self, auxiliary_logits: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = torch.sigmoid(auxiliary_logits.float()).clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )
        entropy = -(
            probabilities * torch.log2(probabilities)
            + (1.0 - probabilities) * torch.log2(1.0 - probabilities)
        )
        routing_mask = StraightThroughEstimator.apply(entropy, self.threshold)
        return routing_mask, entropy
