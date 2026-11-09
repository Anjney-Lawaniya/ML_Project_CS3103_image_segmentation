"""Create training-fold EDA plots and RGB normalization statistics."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

LOGGER = logging.getLogger(__name__)
SEED = 67
CHANNEL_NAMES = ("Red", "Green", "Blue")


def _read_training_filenames(split_path: Path) -> list[str]:
    """Read only the training manifest and return its nonempty filenames."""
    if not split_path.is_file():
        raise FileNotFoundError(f"Training split manifest not found: {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_rgb_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load an image as an RGB float tensor in CHW layout on ``device``."""
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32).div_(255.0)


def _load_binary_mask(mask_path: Path, device: torch.device) -> torch.Tensor:
    """Load a road mask as a 1x1xHxW float tensor on ``device``."""
    with Image.open(mask_path) as mask:
        values = np.asarray(mask).copy()
    if values.ndim == 3:
        values = values[..., 0]
    binary = (values > 0).astype(np.float32, copy=False)
    return torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).to(device=device)


def _plot_class_imbalance(output_dir: Path) -> None:
    """Save the audited background-versus-road class imbalance chart."""
    labels = ("Background", "Road network")
    percentages = (95.23, 4.76)
    colors = ("#64748b", "#e4572e")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bars = axis.bar(labels, percentages, color=colors, width=0.62)
    axis.set_ylabel("Pixels (%)")
    axis.set_ylim(0, 108)
    axis.set_title("Target Class Imbalance")
    axis.grid(axis="y", alpha=0.25)
    for bar, percentage in zip(bars, percentages):
        axis.annotate(
            f"{percentage:.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    figure.savefig(output_dir / "class_imbalance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _calculate_rgb_statistics(
    filenames: Sequence[str], image_dir: Path, output_dir: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate global RGB mean, population standard deviation, and histograms.

    Image tensors and all channel reductions/histogram accumulation run on the
    selected torch device. Only the final small histogram arrays are copied to
    CPU for Matplotlib rendering.
    """
    channel_sum = torch.zeros(3, dtype=torch.float64, device=device)
    channel_square_sum = torch.zeros(3, dtype=torch.float64, device=device)
    channel_histograms = torch.zeros((3, 256), dtype=torch.float64, device=device)
    pixel_count = 0

    for filename in filenames:
        image_tensor = _load_rgb_tensor(image_dir / filename, device)
        image64 = image_tensor.to(dtype=torch.float64)
        channel_sum += image64.sum(dim=(1, 2))
        channel_square_sum += image64.square().sum(dim=(1, 2))
        for channel_index in range(3):
            channel_histograms[channel_index] += torch.histc(
                image_tensor[channel_index], bins=256, min=0.0, max=1.0
            ).to(dtype=torch.float64)
        pixel_count += image_tensor.shape[1] * image_tensor.shape[2]

    if pixel_count == 0:
        raise ValueError("The training manifest contains no images")
    means = channel_sum / pixel_count
    variances = torch.clamp(channel_square_sum / pixel_count - means.square(), min=0.0)
    standard_deviations = variances.sqrt()

    LOGGER.info(
        "Training-fold RGB mean: R=%.8f, G=%.8f, B=%.8f",
        *means.detach().cpu().tolist(),
    )
    LOGGER.info(
        "Training-fold RGB std:  R=%.8f, G=%.8f, B=%.8f",
        *standard_deviations.detach().cpu().tolist(),
    )

    histogram_cpu = channel_histograms.detach().cpu().numpy()
    intensity_centers = np.linspace(0.0, 1.0, 256)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for channel_index, (channel_name, color) in enumerate(
        zip(CHANNEL_NAMES, ("red", "green", "blue"))
    ):
        axis.plot(
            intensity_centers,
            histogram_cpu[channel_index],
            color=color,
            alpha=0.78,
            linewidth=1.25,
            label=channel_name,
        )
    axis.set_xlabel("Normalized pixel intensity")
    axis.set_ylabel("Pixel count")
    axis.set_title("Training-fold RGB Pixel Distribution")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output_dir / "rgb_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return means, standard_deviations


def _plot_boundary_grid(
    filenames: Sequence[str], image_dir: Path, mask_dir: Path, output_dir: Path, device: torch.device
) -> None:
    """Save three seeded random training examples with GPU-computed boundaries."""
    if len(filenames) < 3:
        raise ValueError(f"Need at least three training pairs; found {len(filenames)}")
    sampled_filenames = random.sample(list(filenames), 3)
    laplacian = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
        device=device,
    )
    figure, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)

    for row_index, filename in enumerate(sampled_filenames):
        image_tensor = _load_rgb_tensor(image_dir / filename, device)
        mask_tensor = _load_binary_mask(mask_dir / f"{Path(filename).stem}.tif", device)
        boundary = functional.conv2d(mask_tensor, laplacian, padding=1).abs().squeeze()
        boundary = boundary / boundary.max().clamp_min(1.0)

        image_display = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
        mask_display = mask_tensor.squeeze().detach().cpu().numpy()
        boundary_display = boundary.detach().cpu().numpy()
        axes[row_index, 0].imshow(image_display)
        axes[row_index, 0].set_title(f"RGB Image\n{filename}")
        axes[row_index, 1].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Ground Truth Binary Mask")
        axes[row_index, 2].imshow(boundary_display, cmap="magma", vmin=0, vmax=1)
        axes[row_index, 2].set_title(f"{device.type.upper()} Laplacian Boundary Map")
        for axis in axes[row_index]:
            axis.axis("off")

    figure.savefig(output_dir / "boundary_overlay_grid.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_eda(
    data_dir: str = "data/raw",
    splits_dir: str = "data/splits",
    output_dir: str = "results/plots",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate required EDA figures and RGB statistics from training data only."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("EDA computation device: %s", device)

    filenames = _read_training_filenames(Path(splits_dir) / "train.txt")
    image_dir = Path(data_dir) / "tiff" / "train"
    mask_dir = Path(data_dir) / "tiff" / "train_labels"
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_class_imbalance(plots_dir)
    means, standard_deviations = _calculate_rgb_statistics(
        filenames, image_dir, plots_dir, device
    )
    _plot_boundary_grid(filenames, image_dir, mask_dir, plots_dir, device)
    return means, standard_deviations


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    generate_eda()
