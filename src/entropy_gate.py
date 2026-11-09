"""Differentiable Shannon entropy gates for spatial feature routing."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class StraightThroughEstimator(torch.autograd.Function):
    """Binarize a routing signal while passing gradients through unchanged."""

    @staticmethod
    def forward(ctx: object, continuous_mask: Tensor, threshold: float) -> Tensor:
        return (continuous_mask > threshold).to(dtype=continuous_mask.dtype)

    @staticmethod
    def backward(ctx: object, output_gradient: Tensor) -> tuple[Tensor, None]:
        return output_gradient, None


class ShannonEntropyGate(nn.Module):
    """Produce a binary spatial route from feature-wise pseudo-probabilities."""

    def __init__(self, input_channels: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.probability_projection = nn.Conv2d(input_channels, 2, kernel_size=1)

    def forward(self, feature_map: Tensor) -> Tensor:
        pseudo_probabilities = torch.softmax(self.probability_projection(feature_map), dim=1)
        pixel_entropy = -torch.sum(
            pseudo_probabilities * torch.log(pseudo_probabilities.clamp_min(torch.finfo(feature_map.dtype).eps)),
            dim=1,
            keepdim=True,
        )
        normalized_entropy = pixel_entropy / math.log(2.0)
        return StraightThroughEstimator.apply(normalized_entropy, self.threshold)
