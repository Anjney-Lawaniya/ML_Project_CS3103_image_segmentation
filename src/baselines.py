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


if __name__ == "__main__":
    model = StaticUNet().to("cuda")
    dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
    output = model(dummy_input)
    print(tuple(output.shape))
