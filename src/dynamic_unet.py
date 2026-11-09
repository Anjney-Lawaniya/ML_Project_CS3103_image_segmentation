"""Entropy-routed U-Net with sparse execution in decoder residual blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from entropy_gate import ShannonEntropyGate


class EncoderBlock(nn.Module):
    """Two-layer convolutional encoder block with batch normalization."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_map: Tensor) -> Tensor:
        return self.layers(feature_map)


class MaskedResidualBlock(nn.Module):
    """Apply learned 3x3 residual convolutions only at routed spatial sites."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first_convolution = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.first_normalization = nn.LayerNorm(channels)
        self.second_convolution = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.second_normalization = nn.LayerNorm(channels)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _sparse_convolution(feature_map: Tensor, convolution: nn.Conv2d, active_mask: Tensor) -> Tensor:
        batch_size, _, height, width = feature_map.shape
        unfolded_patches = functional.unfold(
            feature_map,
            kernel_size=convolution.kernel_size,
            dilation=convolution.dilation,
            padding=convolution.padding,
            stride=convolution.stride,
        )
        output_channels = convolution.out_channels
        output_per_batch = []
        for batch_index in range(batch_size):
            active_indices = active_mask[batch_index, 0].reshape(-1).nonzero(as_tuple=False).reshape(-1)
            if active_indices.numel() == 0:
                flattened_output = feature_map.new_zeros((output_channels, height * width))
            else:
                active_patches = unfolded_patches[batch_index].index_select(1, active_indices).transpose(0, 1)
                active_output = functional.linear(
                    active_patches,
                    convolution.weight.flatten(start_dim=1),
                    convolution.bias,
                ).transpose(0, 1)
                flattened_output = feature_map.new_zeros((output_channels, height * width)).index_copy(
                    1, active_indices, active_output
                )
            output_per_batch.append(flattened_output.reshape(output_channels, height, width))
        return torch.stack(output_per_batch, dim=0)

    @staticmethod
    def _normalize_channels(feature_map: Tensor, normalization: nn.LayerNorm) -> Tensor:
        channels_last = feature_map.permute(0, 2, 3, 1)
        return normalization(channels_last).permute(0, 3, 1, 2)

    def forward(self, feature_map: Tensor, routing_mask: Tensor) -> Tensor:
        active_mask = routing_mask.to(dtype=torch.bool)
        first_layer_mask = functional.max_pool2d(
            active_mask.to(dtype=feature_map.dtype), kernel_size=3, stride=1, padding=1
        ).to(dtype=torch.bool)
        first_features = self._sparse_convolution(
            feature_map, self.first_convolution, first_layer_mask
        )
        first_features = self.activation(self._normalize_channels(first_features, self.first_normalization))
        first_features = torch.where(first_layer_mask, first_features, torch.zeros_like(first_features))
        residual_features = self._sparse_convolution(
            first_features, self.second_convolution, active_mask
        )
        residual_features = self.activation(
            self._normalize_channels(residual_features, self.second_normalization)
        )
        return torch.where(active_mask, feature_map + residual_features, feature_map)


class DynamicDecoderStage(nn.Module):
    """Upsample, fuse a skip feature, and entropy-route residual refinement."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(output_channels + skip_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.entropy_gate = ShannonEntropyGate(output_channels)
        self.masked_residual = MaskedResidualBlock(output_channels)

    def forward(self, decoder_feature: Tensor, skip_feature: Tensor) -> Tensor:
        upsampled_feature = self.upsample(decoder_feature)
        fused_feature = self.fusion(torch.cat((upsampled_feature, skip_feature), dim=1))
        routing_mask = self.entropy_gate(fused_feature)
        return self.masked_residual(fused_feature, routing_mask)


class DynamicUNet(nn.Module):
    """Four-stage U-Net with Shannon entropy routing in every decoder stage."""

    def __init__(self, input_channels: int = 3, output_channels: int = 1) -> None:
        super().__init__()
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = EncoderBlock(512, 1024)
        self.decoder4 = DynamicDecoderStage(1024, 512, 512)
        self.decoder3 = DynamicDecoderStage(512, 256, 256)
        self.decoder2 = DynamicDecoderStage(256, 128, 128)
        self.decoder1 = DynamicDecoderStage(128, 64, 64)
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))
        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        return self.output_layer(decoder1)


if __name__ == "__main__":
    model = DynamicUNet().to("cuda")
    dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
    output = model(dummy_input)
    print(tuple(output.shape))
