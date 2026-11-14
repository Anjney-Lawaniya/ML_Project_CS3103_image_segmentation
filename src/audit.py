"""Audit training images and masks and summarize road/background pixel balance."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
IMAGE_DIRECTORY = Path("tiff") / "train"
MASK_DIRECTORY = Path("tiff") / "train_labels"
VALID_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _resolve_pair(raw_dir: Path, filename: str) -> tuple[Path, Path]:
    """Resolve a manifest image filename to its expected image and mask paths."""
    image_path = raw_dir / IMAGE_DIRECTORY / filename
    mask_dir = raw_dir / MASK_DIRECTORY
    mask_candidates = (
        (
            candidate
            for candidate in mask_dir.iterdir()
            if candidate.is_file()
            and candidate.suffix.casefold() in VALID_IMAGE_SUFFIXES
            and candidate.stem.casefold() == image_path.stem.casefold()
        )
        if mask_dir.is_dir()
        else iter(())
    )
    mask_path = next(mask_candidates, mask_dir / f"{image_path.stem}.tif")
    return image_path, mask_path


def audit_training_data(
    data_dir: str = "data/raw",
    splits_dir: str = "data/splits",
    report_path: str = "results/tables/data_audit_summary.txt",
) -> dict[str, int | float]:
    """Audit every listed training pair and save pixel counts and percentages.

    Only ``train.txt`` is read. Validation and held-out test manifests are not
    opened or inspected. Foreground pixels are mask values 1 or 255; background
    pixels are 0. Intermediate mask values are reported and excluded from the
    binary pixel totals.

    Args:
        data_dir: Raw dataset root containing ``tiff/train`` and
            ``tiff/train_labels``.
        splits_dir: Directory containing the generated split manifests.
        report_path: Destination for the plain text audit summary.

    Returns:
        A dictionary containing image, corruption, mask, and pixel counts plus
        background and foreground percentages.

    Raises:
        FileNotFoundError: If the training manifest cannot be found.
    """
    train_manifest = Path(splits_dir) / "train.txt"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Training split manifest not found: {train_manifest}")
    filenames = [
        line.strip()
        for line in train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    raw_dir = Path(data_dir)
    unreadable_files: set[str] = set()
    nonbinary_masks: set[str] = set()
    background_pixels = 0
    foreground_pixels = 0
    effective_background_pixels = 0
    effective_foreground_pixels = 0
    intermediate_mask_pixels = 0

    for filename in tqdm(filenames, desc="Auditing training data", unit="pair"):
        image_path, mask_path = _resolve_pair(raw_dir, filename)
        image_ok = True
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError):
            LOGGER.error("Unreadable or corrupted image: %s", image_path)
            unreadable_files.add(str(image_path))
            image_ok = False

        mask_array: np.ndarray | None = None
        try:
            with Image.open(mask_path) as mask:
                mask.verify()
            with Image.open(mask_path) as mask:
                mask_array = np.asarray(mask)
        except (OSError, ValueError, UnidentifiedImageError):
            LOGGER.error("Unreadable or corrupted mask: %s", mask_path)
            unreadable_files.add(str(mask_path))

        if mask_array is None:
            continue
        allowed = np.isin(mask_array, (0, 1, 255))
        if not bool(np.all(allowed)):
            nonbinary_masks.add(str(mask_path))
            LOGGER.warning("Mask contains intermediate values: %s", mask_path)

        if not image_ok:
            continue
        background_pixels += int(np.count_nonzero(mask_array == 0))
        foreground_pixels += int(np.count_nonzero(np.isin(mask_array, (1, 255))))
        effective_background_pixels += int(np.count_nonzero(mask_array == 0))
        effective_foreground_pixels += int(np.count_nonzero(mask_array > 0))
        intermediate_mask_pixels += int(np.count_nonzero(~allowed))

    binary_pixel_total = background_pixels + foreground_pixels
    if binary_pixel_total:
        background_percentage = 100.0 * background_pixels / binary_pixel_total
        foreground_percentage = 100.0 * foreground_pixels / binary_pixel_total
    else:
        background_percentage = 0.0
        foreground_percentage = 0.0
    effective_pixel_total = effective_background_pixels + effective_foreground_pixels
    if effective_pixel_total:
        effective_background_percentage = 100.0 * effective_background_pixels / effective_pixel_total
        effective_foreground_percentage = 100.0 * effective_foreground_pixels / effective_pixel_total
    else:
        effective_background_percentage = 0.0
        effective_foreground_percentage = 0.0

    results: dict[str, int | float] = {
        "total_images": len(filenames),
        "total_corrupted_files": len(unreadable_files),
        "total_nonbinary_masks": len(nonbinary_masks),
        "nonbinary_masks": len(nonbinary_masks),
        "background_pixels": background_pixels,
        "foreground_pixels": foreground_pixels,
        "intermediate_mask_pixels": intermediate_mask_pixels,
        "background_percentage": background_percentage,
        "foreground_percentage": foreground_percentage,
        "strict_background_percentage": background_percentage,
        "strict_foreground_percentage": foreground_percentage,
        "effective_background_pixels": effective_background_pixels,
        "effective_foreground_pixels": effective_foreground_pixels,
        "effective_background_percentage": effective_background_percentage,
        "effective_foreground_percentage": effective_foreground_percentage,
    }
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "Training Data Audit Summary\n"
        "============================\n"
        f"Total training images: {len(filenames)}\n"
        f"Total unreadable/corrupted files: {len(unreadable_files)}\n"
        "Pixel totals include only pairs with a readable image and mask.\n"
        f"Masks with non-binary values: {len(nonbinary_masks)}\n"
        f"Background pixels (value 0): {background_pixels}\n"
        f"Foreground pixels (value 1 or 255): {foreground_pixels}\n"
        f"Intermediate mask pixels (excluded): {intermediate_mask_pixels}\n"
        f"Strict background: {background_percentage:.12f}% of binary pixels\n"
        f"Strict foreground: {foreground_percentage:.12f}% of binary pixels\n"
        f"Training-effective background pixels (value 0): {effective_background_pixels}\n"
        f"Training-effective foreground pixels (value > 0): {effective_foreground_pixels}\n"
        f"Training-effective background: {effective_background_percentage:.12f}% of pixels\n"
        f"Training-effective foreground: {effective_foreground_percentage:.12f}% of pixels\n",
        encoding="utf-8",
    )
    LOGGER.info("Training data audit report saved to %s", report)
    return results


def main() -> None:
    """Parse command-line paths and run the training-only dataset audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Raw dataset root")
    parser.add_argument("--splits-dir", default="data/splits", help="Split manifest directory")
    parser.add_argument(
        "--report-path",
        default="results/tables/data_audit_summary.txt",
        help="Output text report path",
    )
    args = parser.parse_args()
    audit_training_data(args.data_dir, args.splits_dir, args.report_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
