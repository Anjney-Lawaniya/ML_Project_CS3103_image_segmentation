"""Paired segmentation transforms with training-fold RGB normalization."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as v2_functional

TRAIN_MEAN = (0.43380652, 0.43693942, 0.40113835)
TRAIN_STD = (0.29580833, 0.28946298, 0.30155425)


@dataclass(frozen=True)
class SegmentationTransform:
    """Convert and normalize paired PIL data, optionally augmenting it jointly.

    Spatial operations use torchvision.transforms.v2 functional transforms
    with shared random decisions. Masks use nearest-neighbor interpolation and
    are thresholded after geometry, so their output remains strictly binary.
    """

    training: bool

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform an RGB image and grayscale mask into normalized tensors."""
        image_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32).div_(255.0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(torch.float32)
        mask_tensor = (mask_tensor > 0).to(torch.float32)

        if image_tensor.shape[-2:] != mask_tensor.shape[-2:]:
            raise ValueError(
                "Image and mask dimensions differ: "
                f"{tuple(image_tensor.shape[-2:])} vs {tuple(mask_tensor.shape[-2:])}"
            )

        if self.training:
            if random.random() < 0.5:
                image_tensor = v2_functional.horizontal_flip(image_tensor)
                mask_tensor = v2_functional.horizontal_flip(mask_tensor)
            if random.random() < 0.5:
                image_tensor = v2_functional.vertical_flip(image_tensor)
                mask_tensor = v2_functional.vertical_flip(mask_tensor)
            if random.random() < 0.5:
                quarter_turns = random.choice((1, 2, 3))
                angle = quarter_turns * 90
                image_tensor = v2_functional.rotate(
                    image_tensor,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                )
                mask_tensor = v2_functional.rotate(
                    mask_tensor,
                    angle=angle,
                    interpolation=InterpolationMode.NEAREST,
                )

        mean = torch.tensor(TRAIN_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(TRAIN_STD, dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        mask_tensor = (mask_tensor > 0.5).to(torch.float32)
        return image_tensor.to(torch.float32), mask_tensor.to(torch.float32)


def get_train_transforms() -> SegmentationTransform:
    """Return paired random flips/quarter-turn rotation and normalization."""
    return SegmentationTransform(training=True)


def get_val_test_transforms() -> SegmentationTransform:
    """Return deterministic tensor conversion and training-fold normalization."""
    return SegmentationTransform(training=False)
