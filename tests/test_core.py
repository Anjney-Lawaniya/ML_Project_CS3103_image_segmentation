"""Fast CPU checks for sparse routing, metrics, and decoder FLOPs."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.dynamic_unet import DualPathDecoderBlock
from src.entropy_gate import ShannonEntropyGate
from src.flops_profiler import (
    attention_decoder_gflops,
    dynamic_decoder_gflops,
    static_decoder_gflops,
)
from src.metrics import compute_confusion_counts, compute_segmentation_metrics


class CoreBehaviorTests(unittest.TestCase):
    """Check core tensor and analytical behavior without GPU requirements."""

    def test_sparse_convolution_matches_dense_on_active_sites(self) -> None:
        torch.manual_seed(67)
        inputs = torch.randn(2, 4, 9, 11)
        convolution = nn.Conv2d(4, 5, kernel_size=3, padding=1)
        active_mask = torch.rand(2, 1, 9, 11) > 0.55
        sparse_output = DualPathDecoderBlock._sparse_convolution(
            inputs, convolution, active_mask
        )
        dense_output = convolution(inputs)
        self.assertTrue(
            torch.allclose(
                sparse_output.permute(0, 2, 3, 1)[active_mask[:, 0]],
                dense_output.permute(0, 2, 3, 1)[active_mask[:, 0]],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_zero_threshold_routes_all_entropy_pixels_to_heavy_path(self) -> None:
        torch.manual_seed(67)
        gate = ShannonEntropyGate(threshold=0.0)
        auxiliary_logits = torch.randn(2, 1, 8, 8)
        routing_mask, _ = gate(auxiliary_logits)
        self.assertTrue(torch.all(routing_mask == 1))
        inputs = torch.randn(2, 4, 8, 8)
        decoder_block = DualPathDecoderBlock(channels=4)
        recorded_masks = []
        sparse_convolution = decoder_block._sparse_convolution

        def record_active_mask(feature_map, convolution, active_mask):
            recorded_masks.append(active_mask.detach().clone())
            return sparse_convolution(feature_map, convolution, active_mask)

        decoder_block._sparse_convolution = record_active_mask
        decoder_output = decoder_block(inputs, routing_mask)
        self.assertEqual(len(recorded_masks), 2)
        self.assertTrue(all(torch.all(active_mask) for active_mask in recorded_masks))
        self.assertEqual(tuple(decoder_output.shape), tuple(inputs.shape))

    def test_entropy_gate_passes_entropy_gradient_to_logits(self) -> None:
        auxiliary_logits = torch.tensor([[[[-0.8, 0.3], [0.9, -0.2]]]], requires_grad=True)
        upstream_gradient = torch.tensor([[[[0.2, 0.7], [1.1, -0.4]]]])
        routing_mask, entropy = ShannonEntropyGate(threshold=0.5)(auxiliary_logits)
        routing_mask.backward(upstream_gradient, retain_graph=True)
        expected_gradient = torch.autograd.grad(
            entropy, auxiliary_logits, grad_outputs=upstream_gradient
        )[0]
        self.assertTrue(torch.all(auxiliary_logits.grad.abs() > 0))
        self.assertTrue(torch.allclose(auxiliary_logits.grad, expected_gradient, atol=1e-6))

    def test_segmentation_metrics_empty_and_known_counts(self) -> None:
        empty_logits = torch.full((1, 1, 2, 2), -10.0)
        empty_targets = torch.zeros_like(empty_logits)
        self.assertEqual(
            compute_segmentation_metrics(empty_logits, empty_targets),
            {"iou": 1.0, "f1_score": 1.0, "precision": 1.0, "recall": 1.0},
        )
        logits = torch.tensor([[[[10.0, 10.0], [-10.0, 10.0]]]])
        targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        self.assertEqual(
            compute_confusion_counts(logits, targets),
            {"true_positives": 1, "false_positives": 2, "false_negatives": 1, "true_negatives": 0},
        )

    def test_decoder_flops_sanity(self) -> None:
        self.assertAlmostEqual(static_decoder_gflops(), 249.142, places=3)
        self.assertAlmostEqual(attention_decoder_gflops(), 257.763, places=3)
        self.assertLess(dynamic_decoder_gflops([1.0, 1.0, 1.0]), static_decoder_gflops())


if __name__ == "__main__":
    unittest.main()
