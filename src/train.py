"""Command-line training pipeline for road segmentation architectures."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from src.baselines import SoftAttentionUNet, StaticUNet
from src.dataloaders import get_dataloaders
from src.dynamic_unet import DynamicUNet
from src.losses import BCEDiceLoss, DynamicUNetLoss
from src.metrics import compute_confusion_counts, metrics_from_counts

RANDOM_SEED = 67
LOGGER = logging.getLogger(__name__)


def _seed_everything(seed: int = RANDOM_SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _build_model(model_name: str, threshold: float) -> nn.Module:
    """Construct the requested baseline or dynamic U-Net model."""
    if model_name == "static_unet":
        return StaticUNet()
    if model_name == "attention_unet":
        return SoftAttentionUNet()
    if model_name == "dynamic_unet":
        return DynamicUNet(threshold=threshold)
    raise ValueError(f"Unsupported model: {model_name}")


def _forward_with_loss(
    model: nn.Module,
    criterion: BCEDiceLoss | DynamicUNetLoss,
    inputs: Tensor,
    targets: Tensor,
    is_dynamic_model: bool,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    """Run a model and its matching segmentation objective."""
    if is_dynamic_model:
        logits, auxiliary_logits_list, routing_masks_list = model(
            inputs,
            return_internals=True,
        )
        loss = criterion(
            logits,
            targets,
            auxiliary_logits_list,
            routing_masks_list,
        )
        return logits, loss, auxiliary_logits_list, routing_masks_list
    logits = model(inputs)
    loss = criterion(logits, targets)
    return logits, loss, [], []


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: BCEDiceLoss | DynamicUNetLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch_index: int,
    epoch_count: int,
    is_dynamic_model: bool,
    accumulation_steps: int,
) -> float:
    """Train for one epoch and return the mean sample-weighted loss."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    model.train()
    total_loss = 0.0
    sample_count = 0
    total_batches = len(data_loader)
    progress_bar = tqdm(
        data_loader,
        desc=f"Train {epoch_index}/{epoch_count}",
        leave=False,
    )
    for batch_index, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _, loss, _, _ = _forward_with_loss(
                model,
                criterion,
                inputs,
                targets,
                is_dynamic_model,
            )
        accumulated_loss = loss / accumulation_steps
        scaler.scale(accumulated_loss).backward()
        is_accumulation_boundary = (batch_index + 1) % accumulation_steps == 0
        is_last_batch = batch_index + 1 == total_batches
        if is_accumulation_boundary or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = inputs.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: BCEDiceLoss | DynamicUNetLoss,
    device: torch.device,
    epoch_index: int,
    epoch_count: int,
    is_dynamic_model: bool,
) -> dict[str, float]:
    """Evaluate loss, segmentation metrics, and dynamic routing utilization."""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    pooled_counts = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
    }
    macro_f1_total = 0.0
    active_percentage_total = 0.0
    progress_bar = tqdm(
        data_loader,
        desc=f"Valid {epoch_index}/{epoch_count}",
        leave=False,
    )
    for inputs, targets in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, loss, _, routing_masks_list = _forward_with_loss(
                model,
                criterion,
                inputs,
                targets,
                is_dynamic_model,
            )
        batch_size = inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        batch_counts = compute_confusion_counts(logits.float(), targets.float())
        for metric_name, metric_value in batch_counts.items():
            pooled_counts[metric_name] += metric_value
        macro_f1_total += metrics_from_counts(batch_counts)["f1_score"] * batch_size
        if routing_masks_list:
            batch_active_percentage = 100.0 * torch.stack(
                [routing_mask.float().mean() for routing_mask in routing_masks_list]
            ).mean()
            active_percentage_total += float(batch_active_percentage.item()) * batch_size
        sample_count += batch_size
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    denominator = max(sample_count, 1)
    results = {
        "loss": total_loss / denominator,
        **metrics_from_counts(pooled_counts),
        "macro_f1_score": macro_f1_total / denominator,
        "active_percentage": active_percentage_total / denominator,
    }
    return results


def _append_epoch_history(
    log_path: Path, row: dict[str, object], overwrite: bool = False
) -> None:
    """Append one epoch's metrics to the model-specific CSV history."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = overwrite or not log_path.exists()
    mode = "w" if overwrite else "a"
    with log_path.open(mode, newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options for a training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("static_unet", "attention_unet", "dynamic_unet"),
        default="dynamic_unet",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("--flops_weight", "--flops-weight", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", "--accumulation-steps", type=int, default=4)
    parsed_arguments = parser.parse_args(arguments)
    if parsed_arguments.epochs <= 0:
        parser.error("--epochs must be positive")
    if parsed_arguments.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if parsed_arguments.learning_rate <= 0.0:
        parser.error("--learning_rate must be positive")
    if parsed_arguments.flops_weight < 0.0:
        parser.error("--flops_weight must be nonnegative")
    if parsed_arguments.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    if parsed_arguments.accumulation_steps <= 0:
        parser.error("--accumulation_steps must be positive")
    if not 0.0 <= parsed_arguments.threshold <= 1.0:
        parser.error("--threshold must be in the range [0, 1]")
    return parsed_arguments


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the configured training and validation epochs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    options = parse_args(arguments)
    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, _ = get_dataloaders(
        batch_size=options.batch_size,
        num_workers=options.num_workers,
    )
    model = _build_model(options.model, options.threshold).to(device)
    is_dynamic_model = options.model == "dynamic_unet"
    if is_dynamic_model:
        criterion: BCEDiceLoss | DynamicUNetLoss = DynamicUNetLoss(
            flops_weight=options.flops_weight
        )
    else:
        criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{options.model}.pth"
    history_path = Path("results") / "tables" / f"{options.model}_training_log.csv"
    best_validation_f1 = float("-inf")

    LOGGER.info("Training %s on %s", options.model, device)
    for epoch_index in range(1, options.epochs + 1):
        training_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch_index,
            options.epochs,
            is_dynamic_model,
            options.accumulation_steps,
        )
        validation_results = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            epoch_index,
            options.epochs,
            is_dynamic_model,
        )
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | IoU=%.6f | F1=%.6f | precision=%.6f | recall=%.6f%s",
            epoch_index,
            options.epochs,
            training_loss,
            validation_results["loss"],
            validation_results["iou"],
            validation_results["f1_score"],
            validation_results["precision"],
            validation_results["recall"],
            f" | active_3x3={validation_results['active_percentage']:.2f}%"
            if is_dynamic_model
            else "",
        )
        history_row: dict[str, object] = {
            "epoch": epoch_index,
            "train_loss": training_loss,
            "validation_loss": validation_results["loss"],
            "validation_iou": validation_results["iou"],
            "validation_f1": validation_results["f1_score"],
            "validation_macro_f1": validation_results["macro_f1_score"],
            "validation_precision": validation_results["precision"],
            "validation_recall": validation_results["recall"],
            "active_3x3_percentage": validation_results["active_percentage"]
            if is_dynamic_model
            else "",
        }
        _append_epoch_history(history_path, history_row, overwrite=epoch_index == 1)
        if validation_results["f1_score"] > best_validation_f1:
            best_validation_f1 = validation_results["f1_score"]
            torch.save(
                {
                    "epoch": epoch_index,
                    "model_name": options.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_results,
                    "random_seed": RANDOM_SEED,
                    "threshold": options.threshold,
                    "flops_weight": options.flops_weight,
                    "cli_args": vars(options),
                },
                checkpoint_path,
            )
            LOGGER.info("Saved best checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
