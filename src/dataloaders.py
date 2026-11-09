"""Dataset and DataLoader construction for leakage-free road segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.transforms import get_train_transforms, get_val_test_transforms

PairTransform = Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]]
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
RANDOM_SEED = 67


class MassachusettsRoadsDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load RGB image and binary road-mask pairs listed in a split manifest.

    Manifest entries contain image filenames. Images are read from
    ``raw_data_root/tiff/train`` and masks are matched by stem in
    ``raw_data_root/tiff/train_labels``.
    """

    def __init__(
        self,
        split_manifest: str | Path,
        raw_data_root: str | Path,
        transforms: PairTransform | None = None,
    ) -> None:
        """Initialize manifest entries, raw paths, and an optional pair transform."""
        self.split_manifest = Path(split_manifest)
        self.raw_data_root = Path(raw_data_root)
        self.image_dir = self.raw_data_root / "tiff" / "train"
        self.mask_dir = self.raw_data_root / "tiff" / "train_labels"
        self.transforms = transforms

        if not self.split_manifest.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.split_manifest}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.filenames = [
            line.strip()
            for line in self.split_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        masks_by_stem: dict[str, Path] = {}
        for path in self.mask_dir.iterdir():
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
                key = path.stem.casefold()
                if key in masks_by_stem:
                    raise ValueError(f"Multiple masks share the same filename stem: {path.stem}")
                masks_by_stem[key] = path

        self.samples: list[tuple[Path, Path]] = []
        for filename in self.filenames:
            image_path = self.image_dir / filename
            mask_path = masks_by_stem.get(Path(filename).stem.casefold())
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in manifest not found: {image_path}")
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for image listed in manifest: {filename}")
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        """Return the number of image-mask pairs in this manifest."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load, transform, and return one normalized image/mask tensor pair."""
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image, mask)
        else:
            image_array = np.asarray(image, dtype=np.uint8).copy()
            mask_array = np.asarray(mask, dtype=np.uint8).copy()
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32).div_(255.0)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(torch.float32)
            mask_tensor = (mask_tensor > 0).to(torch.float32)

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError(f"Expected image tensor [3,H,W], got {tuple(image_tensor.shape)}")
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError(f"Expected mask tensor [1,H,W], got {tuple(mask_tensor.shape)}")
        mask_tensor = (mask_tensor > 0.5).to(torch.float32)
        return image_tensor.to(torch.float32), mask_tensor


def get_dataloaders(
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create train, validation, and test loaders from the three manifests.

    The training loader shuffles with a fixed generator seed. Validation and
    test loaders preserve manifest order. All three use only their own manifest
    and receive the appropriate paired transforms.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    split_dir = Path("data") / "splits"
    raw_data_root = Path("data") / "raw"
    train_dataset = MassachusettsRoadsDataset(
        split_dir / "train.txt", raw_data_root, get_train_transforms()
    )
    validation_dataset = MassachusettsRoadsDataset(
        split_dir / "val.txt", raw_data_root, get_val_test_transforms()
    )
    test_dataset = MassachusettsRoadsDataset(
        split_dir / "test.txt", raw_data_root, get_val_test_transforms()
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **common_loader_options
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_options)
    return train_loader, validation_loader, test_loader
