"""Static segmentation baselines for satellite road extraction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Two consecutive convolution, batch-normalization, and ReLU layers."""

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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class StaticUNet(nn.Module):
    """Four-stage encoder-decoder U-Net that returns one-channel logits."""

    def __init__(self, input_channels: int = 3, output_channels: int = 1) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(input_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))

        decoder4 = self.up4(bottleneck)
        decoder4 = self.decoder4(torch.cat((decoder4, encoder4), dim=1))
        decoder3 = self.up3(decoder4)
        decoder3 = self.decoder3(torch.cat((decoder3, encoder3), dim=1))
        decoder2 = self.up2(decoder3)
        decoder2 = self.decoder2(torch.cat((decoder2, encoder2), dim=1))
        decoder1 = self.up1(decoder2)
        decoder1 = self.decoder1(torch.cat((decoder1, encoder1), dim=1))
        return self.output_layer(decoder1)


class AttentionGate(nn.Module):
    """Learn a spatial coefficient map for filtering an encoder skip feature."""

    def __init__(self, gating_channels: int, skip_channels: int, intermediate_channels: int) -> None:
        super().__init__()
        self.weight_g = nn.Conv2d(gating_channels, intermediate_channels, kernel_size=1, bias=True)
        self.weight_x = nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gating_signal: Tensor, skip_feature: Tensor) -> Tensor:
        attention_coefficients = self.sigmoid(
            self.psi(self.relu(self.weight_g(gating_signal) + self.weight_x(skip_feature)))
        )
        return skip_feature * attention_coefficients


class SoftAttentionUNet(nn.Module):
    """Four-stage U-Net that soft-attends to encoder features before fusion."""

    def __init__(self, input_channels: int = 3, output_channels: int = 1) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(input_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention4 = AttentionGate(512, 512, 256)
        self.decoder4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention3 = AttentionGate(256, 256, 128)
        self.decoder3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attention2 = AttentionGate(128, 128, 64)
        self.decoder2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention1 = AttentionGate(64, 64, 32)
        self.decoder1 = ConvBlock(128, 64)
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))

        decoder4 = self.up4(bottleneck)
        attended4 = self.attention4(decoder4, encoder4)
        decoder4 = self.decoder4(torch.cat((decoder4, attended4), dim=1))
        decoder3 = self.up3(decoder4)
        attended3 = self.attention3(decoder3, encoder3)
        decoder3 = self.decoder3(torch.cat((decoder3, attended3), dim=1))
        decoder2 = self.up2(decoder3)
        attended2 = self.attention2(decoder2, encoder2)
        decoder2 = self.decoder2(torch.cat((decoder2, attended2), dim=1))
        decoder1 = self.up1(decoder2)
        attended1 = self.attention1(decoder1, encoder1)
        decoder1 = self.decoder1(torch.cat((decoder1, attended1), dim=1))
        return self.output_layer(decoder1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StaticUNet().to(device)
    dummy_input = torch.randn(1, 3, 512, 512, device=device)
    output = model(dummy_input)
    print(tuple(output.shape))
