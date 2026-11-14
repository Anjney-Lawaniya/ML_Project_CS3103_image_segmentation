"""Dynamic U-Net with entropy-gated skip pruning and dual-path decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.entropy_gate import ShannonEntropyGate


class EncoderBlock(nn.Module):
    """Two convolutional layers with batch normalization and ReLU activation."""

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


class DualPathDecoderBlock(nn.Module):
    """Route active sites through sparse 3x3 layers and others through 1x1."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.high_entropy_convolution1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.high_entropy_normalization1 = nn.LayerNorm(channels)
        self.high_entropy_convolution2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.high_entropy_normalization2 = nn.LayerNorm(channels)
        self.low_entropy_projection = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.low_entropy_normalization = nn.LayerNorm(channels)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _sparse_convolution(
        feature_map: Tensor,
        convolution: nn.Conv2d,
        active_mask: Tensor,
    ) -> Tensor:
        batch_size, _, height, width = feature_map.shape
        unfolded_patches = functional.unfold(
            feature_map,
            kernel_size=convolution.kernel_size,
            dilation=convolution.dilation,
            padding=convolution.padding,
            stride=convolution.stride,
        )
        flattened_outputs = []
        for batch_index in range(batch_size):
            active_indices = active_mask[batch_index, 0].reshape(-1).nonzero(as_tuple=False).reshape(-1)
            if active_indices.numel() == 0:
                flattened_output = feature_map.new_zeros(
                    (convolution.out_channels, height * width)
                )
            else:
                active_patches = unfolded_patches[batch_index].index_select(
                    1, active_indices
                ).transpose(0, 1)
                active_values = functional.linear(
                    active_patches,
                    convolution.weight.flatten(start_dim=1),
                    convolution.bias,
                ).transpose(0, 1)
                flattened_output = feature_map.new_zeros(
                    (convolution.out_channels, height * width)
                ).index_copy(1, active_indices, active_values.to(feature_map.dtype))
            flattened_outputs.append(
                flattened_output.reshape(convolution.out_channels, height, width)
            )
        return torch.stack(flattened_outputs, dim=0)

    @staticmethod
    def _normalize_channels(feature_map: Tensor, normalization: nn.LayerNorm) -> Tensor:
        channels_last = feature_map.permute(0, 2, 3, 1)
        return normalization(channels_last).permute(0, 3, 1, 2)

    def forward(self, fused_feature: Tensor, routing_mask: Tensor) -> Tensor:
        active_mask = routing_mask.to(dtype=torch.bool)
        active_neighborhood = functional.max_pool2d(
            active_mask.to(dtype=fused_feature.dtype), kernel_size=3, stride=1, padding=1
        ).to(dtype=torch.bool)
        high_entropy_features = self._sparse_convolution(
            fused_feature,
            self.high_entropy_convolution1,
            active_neighborhood,
        )
        high_entropy_features = self.activation(
            self._normalize_channels(
                high_entropy_features,
                self.high_entropy_normalization1,
            )
        )
        high_entropy_features = torch.where(
            active_neighborhood,
            high_entropy_features,
            torch.zeros_like(high_entropy_features),
        )
        high_entropy_features = self._sparse_convolution(
            high_entropy_features,
            self.high_entropy_convolution2,
            active_mask,
        )
        high_entropy_features = self.activation(
            self._normalize_channels(
                high_entropy_features,
                self.high_entropy_normalization2,
            )
        )
        low_entropy_features = self.low_entropy_projection(fused_feature)
        low_entropy_features = self.activation(
            self._normalize_channels(
                low_entropy_features,
                self.low_entropy_normalization,
            )
        )
        return torch.where(
            active_mask,
            fused_feature + high_entropy_features,
            low_entropy_features,
        )


class DenseDecoderStage(nn.Module):
    """Upsample and densely fuse the coarsest decoder skip feature."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.fusion = EncoderBlock(output_channels + skip_channels, output_channels)

    def forward(self, decoder_feature: Tensor, skip_feature: Tensor) -> Tensor:
        upsampled_feature = self.upsample(decoder_feature)
        return self.fusion(torch.cat((upsampled_feature, skip_feature), dim=1))


class GatedDecoderStage(nn.Module):
    """Predict entropy, prune a skip, fuse features, and execute both lanes."""

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        threshold: float,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.auxiliary_head = nn.Conv2d(output_channels, 1, kernel_size=1)
        self.entropy_gate = ShannonEntropyGate(threshold=threshold)
        self.fusion = nn.Sequential(
            nn.Conv2d(output_channels + skip_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder_block = DualPathDecoderBlock(output_channels)

    def forward(
        self,
        decoder_feature: Tensor,
        skip_feature: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        upsampled_feature = self.upsample(decoder_feature)
        auxiliary_logits = self.auxiliary_head(upsampled_feature)
        routing_mask, _ = self.entropy_gate(auxiliary_logits)
        pruned_skip = skip_feature * routing_mask.to(dtype=skip_feature.dtype)
        fused_feature = self.fusion(
            torch.cat((upsampled_feature, pruned_skip), dim=1)
        )
        routed_feature = self.decoder_block(fused_feature, routing_mask)
        return routed_feature, auxiliary_logits, routing_mask


class DynamicUNet(nn.Module):
    """Four-stage U-Net with deep-supervised entropy-routed fine decoders."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = EncoderBlock(512, 1024)
        self.decoder4 = DenseDecoderStage(1024, 512, 512)
        self.decoder3 = GatedDecoderStage(512, 256, 256, self.threshold)
        self.decoder2 = GatedDecoderStage(256, 128, 128, self.threshold)
        self.decoder1 = GatedDecoderStage(128, 64, 64, self.threshold)
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def set_entropy_threshold(self, threshold: float) -> None:
        """Update the threshold used by every internal Shannon entropy gate."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Entropy threshold must be in the range [0, 1]")
        self.threshold = float(threshold)
        for module in self.modules():
            if isinstance(module, ShannonEntropyGate):
                module.threshold = self.threshold

    def forward(
        self,
        inputs: Tensor,
        return_internals: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor], list[Tensor]]:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))
        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3, auxiliary_logits3, routing_mask3 = self.decoder3(decoder4, encoder3)
        decoder2, auxiliary_logits2, routing_mask2 = self.decoder2(decoder3, encoder2)
        decoder1, auxiliary_logits1, routing_mask1 = self.decoder1(decoder2, encoder1)
        logits = self.output_layer(decoder1)
        if return_internals:
            return (
                logits,
                [auxiliary_logits3, auxiliary_logits2, auxiliary_logits1],
                [routing_mask3, routing_mask2, routing_mask1],
            )
        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DynamicUNet().to(device)
    dummy_input = torch.randn(1, 3, 512, 512, device=device)
    output = model(dummy_input)
    print(tuple(output.shape))
