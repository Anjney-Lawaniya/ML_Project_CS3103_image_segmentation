"""Reusable inference metrics, threshold sweeps, and routing policy studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader

from src.dynamic_unet import DynamicUNet
from src.flops_profiler import compute_dilated_routing_mask, dynamic_decoder_gflops
from src.metrics import compute_confusion_counts, metrics_from_counts

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


def extract_binary_boundary(binary_mask: Tensor, radius: int = 1) -> Tensor:
    """Extract a binary morphological-gradient boundary using replicate padding."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    binary_mask = binary_mask.float()
    if radius == 0:
        return torch.zeros_like(binary_mask, dtype=torch.bool)
    kernel_size = 2 * radius + 1
    padded_mask = functional.pad(
        binary_mask, (radius, radius, radius, radius), mode="replicate"
    )
    dilated_mask = functional.max_pool2d(padded_mask, kernel_size, stride=1)
    eroded_mask = -functional.max_pool2d(-padded_mask, kernel_size, stride=1)
    return dilated_mask > eroded_mask


def compute_boundary_f1(
    logits: Tensor,
    targets: Tensor,
    boundary_radius: int = 1,
    tolerance_radius: int = 2,
) -> float:
    """Compute symmetric boundary F1 with a configurable pixel matching band."""
    predicted_boundary = extract_binary_boundary(torch.sigmoid(logits) >= 0.5, boundary_radius)
    target_boundary = extract_binary_boundary(targets >= 0.5, boundary_radius)
    predicted_count = int(predicted_boundary.sum().item())
    target_count = int(target_boundary.sum().item())
    if predicted_count == 0 and target_count == 0:
        return 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0
    kernel_size = 2 * tolerance_radius + 1
    predicted_neighborhood = functional.max_pool2d(
        predicted_boundary.float(), kernel_size, stride=1, padding=tolerance_radius
    ) > 0
    target_neighborhood = functional.max_pool2d(
        target_boundary.float(), kernel_size, stride=1, padding=tolerance_radius
    ) > 0
    matched_predictions = int((predicted_boundary & target_neighborhood).sum().item())
    matched_targets = int((target_boundary & predicted_neighborhood).sum().item())
    precision = matched_predictions / predicted_count
    recall = matched_targets / target_count
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _directed_distances_torch(
    source_coordinates: Tensor, destination_coordinates: Tensor, chunk_size: int = 1024
) -> Tensor:
    """Compute chunked nearest-neighbor distances without a full pairwise matrix."""
    distance_chunks = []
    for chunk_start in range(0, source_coordinates.shape[0], chunk_size):
        coordinate_chunk = source_coordinates[chunk_start : chunk_start + chunk_size]
        distance_chunks.append(torch.cdist(coordinate_chunk, destination_coordinates).min(dim=1).values)
    return torch.cat(distance_chunks)


def compute_hd95(
    logits: Tensor, targets: Tensor, boundary_radius: int = 1
) -> tuple[float, bool]:
    """Compute symmetric 95th-percentile boundary distance and validity flag.

    The boolean is true only when both predicted and target boundaries are
    nonempty. Empty-set cases receive a finite diagonal-distance penalty (or
    zero when both are empty) and are excluded from valid HD95 summaries.
    """
    predicted_boundary = extract_binary_boundary(
        torch.sigmoid(logits) >= 0.5, boundary_radius
    )
    target_boundary = extract_binary_boundary(targets >= 0.5, boundary_radius)
    per_image_distances: list[float] = []
    all_valid = True
    for predicted_mask, target_mask in zip(predicted_boundary, target_boundary):
        predicted_coordinates = torch.nonzero(predicted_mask[0], as_tuple=False).float()
        target_coordinates = torch.nonzero(target_mask[0], as_tuple=False).float()
        if not predicted_coordinates.numel() and not target_coordinates.numel():
            per_image_distances.append(0.0)
            all_valid = False
            continue
        if not predicted_coordinates.numel() or not target_coordinates.numel():
            height, width = predicted_mask.shape[-2:]
            per_image_distances.append(float((height**2 + width**2) ** 0.5))
            all_valid = False
            continue
        if distance_transform_edt is not None:
            predicted_array = predicted_mask[0].detach().cpu().numpy().astype(bool)
            target_array = target_mask[0].detach().cpu().numpy().astype(bool)
            predicted_to_target = distance_transform_edt(~target_array)[predicted_array]
            target_to_predicted = distance_transform_edt(~predicted_array)[target_array]
            distances = np.concatenate((predicted_to_target, target_to_predicted))
            per_image_distances.append(float(np.percentile(distances, 95)))
        else:
            predicted_to_target = _directed_distances_torch(
                predicted_coordinates, target_coordinates
            )
            target_to_predicted = _directed_distances_torch(
                target_coordinates, predicted_coordinates
            )
            distances = torch.cat((predicted_to_target, target_to_predicted))
            per_image_distances.append(float(torch.quantile(distances, 0.95).item()))
    return (float(np.mean(per_image_distances)) if per_image_distances else 0.0, all_valid)


def bootstrap_confidence_interval(
    values: Sequence[float],
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 67,
) -> tuple[float, float, float]:
    """Return the sample mean and seeded percentile bootstrap confidence bounds."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        return float("nan"), float("nan"), float("nan")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, sample.size, size=(num_bootstrap, sample.size))
    means = sample[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return float(sample.mean()), float(lower), float(upper)


def _update_counts(total: dict[str, int], batch: dict[str, int]) -> None:
    """Accumulate the four confusion count fields in place."""
    for key, value in batch.items():
        total[key] = total.get(key, 0) + value


def _mask_counts(
    routing_masks: Sequence[Tensor],
) -> tuple[list[float], list[int], list[float], list[int]]:
    """Return raw and dilated active counts and pixel totals per stage."""
    raw_counts = [float(mask.float().sum().item()) for mask in routing_masks]
    raw_totals = [mask.numel() for mask in routing_masks]
    dilated_masks = [compute_dilated_routing_mask(mask) for mask in routing_masks]
    dilated_counts = [float(mask.float().sum().item()) for mask in dilated_masks]
    dilated_totals = [mask.numel() for mask in dilated_masks]
    return raw_counts, raw_totals, dilated_counts, dilated_totals


@torch.no_grad()
def evaluate_threshold_sweep(
    model: DynamicUNet,
    data_loader: DataLoader,
    thresholds: Sequence[float],
    device: torch.device | str,
) -> list[dict[str, float]]:
    """Evaluate micro metrics and nominal/halo FLOPs across gate thresholds."""
    device = torch.device(device)
    original_threshold = model.threshold
    original_training_state = model.training
    results: list[dict[str, float]] = []
    try:
        model.eval()
        for threshold in thresholds:
            model.set_entropy_threshold(float(threshold))
            counts = {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "true_negatives": 0}
            raw_stage_counts: list[float] | None = None
            raw_stage_totals: list[int] | None = None
            halo_stage_counts: list[float] | None = None
            halo_stage_totals: list[int] | None = None
            for inputs, targets in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits, _, routing_masks = model(inputs, return_internals=True)
                _update_counts(counts, compute_confusion_counts(logits.float(), targets.float()))
                raw_counts, raw_totals, halo_counts, halo_totals = _mask_counts(routing_masks)
                if raw_stage_counts is None:
                    raw_stage_counts = [0.0] * len(raw_counts)
                    raw_stage_totals = [0] * len(raw_totals)
                    halo_stage_counts = [0.0] * len(halo_counts)
                    halo_stage_totals = [0] * len(halo_totals)
                for stage_index in range(len(raw_counts)):
                    raw_stage_counts[stage_index] += raw_counts[stage_index]
                    raw_stage_totals[stage_index] += raw_totals[stage_index]
                    halo_stage_counts[stage_index] += halo_counts[stage_index]
                    halo_stage_totals[stage_index] += halo_totals[stage_index]
            raw_fractions = [
                count / total if total else 0.0
                for count, total in zip(raw_stage_counts or [], raw_stage_totals or [])
            ]
            halo_fractions = [
                count / total if total else 0.0
                for count, total in zip(halo_stage_counts or [], halo_stage_totals or [])
            ]
            metrics = metrics_from_counts(counts)
            results.append(
                {
                    "threshold": float(threshold),
                    **metrics,
                    "raw_active_percentage": 100.0 * float(np.mean(raw_fractions)) if raw_fractions else 0.0,
                    "dilated_halo_active_percentage": 100.0 * float(np.mean(halo_fractions)) if halo_fractions else 0.0,
                    "nominal_decoder_gflops": dynamic_decoder_gflops(raw_fractions),
                    "halo_adjusted_decoder_gflops": dynamic_decoder_gflops(
                        raw_fractions,
                        dilated_active_fractions=halo_fractions,
                        conditional_low_entropy=False,
                    ),
                }
            )
    finally:
        model.set_entropy_threshold(original_threshold)
        model.train(original_training_state)
    return results


def _policy_hook(stage_index: int, state: dict[str, Any], seed: int):
    """Build a per-sample random/inverted route override hook for one gate."""
    def apply_policy(module: torch.nn.Module, module_inputs: tuple[Any, ...], module_outputs: Any):
        """Replace gate masks with per-sample budget-matched alternatives."""
        routing_mask, entropy_map = module_outputs
        policy = state["policy"]
        if policy not in ("random", "inverted"):
            return module_outputs
        replacement_mask = torch.zeros_like(routing_mask)
        batch_size, _, height, width = routing_mask.shape
        for sample_index in range(batch_size):
            active_count = int(routing_mask[sample_index, 0].sum().item())
            flattened_mask = replacement_mask[sample_index, 0].reshape(-1)
            sample_counter = state["sample_counters"][stage_index]
            state["sample_counters"][stage_index] += 1
            if active_count == 0:
                continue
            if policy == "random":
                generator = torch.Generator(device=routing_mask.device)
                generator.manual_seed(
                    seed + sample_counter * 31 + stage_index
                )
                selected_indices = torch.randperm(
                    height * width, generator=generator, device=routing_mask.device
                )[:active_count]
            else:
                flattened_entropy = entropy_map[sample_index, 0].reshape(-1)
                selected_indices = torch.topk(
                    flattened_entropy, k=active_count, largest=False
                ).indices
            flattened_mask[selected_indices] = 1.0
        return replacement_mask, entropy_map

    return apply_policy


@torch.no_grad()
def evaluate_routing_policies(
    model: DynamicUNet,
    data_loader: DataLoader,
    device: torch.device | str,
    threshold: float = 0.5,
    seed: int = 67,
) -> list[tuple[str, dict[str, float]]]:
    """Compare entropy, per-tile matched random/inverted, bypass, and all-active routes."""
    device = torch.device(device)
    original_threshold = model.threshold
    original_training_state = model.training
    gate_modules = (
        model.decoder3.entropy_gate,
        model.decoder2.entropy_gate,
        model.decoder1.entropy_gate,
    )
    policy_specs = (
        ("Shannon Entropy Routing (tau=0.5)", "entropy", threshold),
        ("Random Bernoulli Routing (Per-Tile Matched Budget)", "random", threshold),
        ("Inverted Confidence Routing (Bottom-k Lowest Entropy, Per-Tile Matched Budget)", "inverted", threshold),
        ("Pure 1x1 Bypass (tau=1.0, 0% Active 3x3)", "entropy", 1.0),
        ("All-Active Dynamic U-Net (tau=0.0, 100% Active 3x3)", "entropy", 0.0),
    )
    results: list[tuple[str, dict[str, float]]] = []
    hook_handles = []
    try:
        model.eval()
        for display_name, policy_name, policy_threshold in policy_specs:
            state: dict[str, Any] = {
                "policy": policy_name,
                "sample_counters": [0, 0, 0],
            }
            if policy_name in ("random", "inverted"):
                hook_handles = [
                    gate.register_forward_hook(_policy_hook(index, state, seed))
                    for index, gate in enumerate(gate_modules)
                ]
            model.set_entropy_threshold(policy_threshold)
            counts = {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "true_negatives": 0}
            raw_stage_counts: list[float] | None = None
            raw_stage_totals: list[int] | None = None
            halo_stage_counts: list[float] | None = None
            halo_stage_totals: list[int] | None = None
            for inputs, targets in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits, _, routing_masks = model(inputs, return_internals=True)
                _update_counts(counts, compute_confusion_counts(logits.float(), targets.float()))
                raw_counts, raw_totals, halo_counts, halo_totals = _mask_counts(routing_masks)
                if raw_stage_counts is None:
                    raw_stage_counts = [0.0] * len(raw_counts)
                    raw_stage_totals = [0] * len(raw_totals)
                    halo_stage_counts = [0.0] * len(halo_counts)
                    halo_stage_totals = [0] * len(halo_totals)
                for stage_index in range(len(raw_counts)):
                    raw_stage_counts[stage_index] += raw_counts[stage_index]
                    raw_stage_totals[stage_index] += raw_totals[stage_index]
                    halo_stage_counts[stage_index] += halo_counts[stage_index]
                    halo_stage_totals[stage_index] += halo_totals[stage_index]
            for handle in hook_handles:
                handle.remove()
            hook_handles = []
            raw_fractions = [
                count / total if total else 0.0
                for count, total in zip(raw_stage_counts or [], raw_stage_totals or [])
            ]
            halo_fractions = [
                count / total if total else 0.0
                for count, total in zip(halo_stage_counts or [], halo_stage_totals or [])
            ]
            metrics = metrics_from_counts(counts)
            metrics.update(
                {
                    "raw_active_percentage": 100.0 * float(np.mean(raw_fractions)) if raw_fractions else 0.0,
                    "dilated_halo_active_percentage": 100.0 * float(np.mean(halo_fractions)) if halo_fractions else 0.0,
                    "nominal_decoder_gflops": dynamic_decoder_gflops(raw_fractions),
                    "halo_adjusted_decoder_gflops": dynamic_decoder_gflops(
                        raw_fractions,
                        dilated_active_fractions=halo_fractions,
                        conditional_low_entropy=False,
                    ),
                }
            )
            results.append((display_name, metrics))
    finally:
        for handle in hook_handles:
            handle.remove()
        model.set_entropy_threshold(original_threshold)
        model.train(original_training_state)
    return results
