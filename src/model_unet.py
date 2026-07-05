import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two consecutive convolution layers with BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block: MaxPool followed by ConvBlock.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block followed by skip connection and ConvBlock.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Fix possible size mismatch caused by odd image sizes
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    """
    Lightweight U-Net for image enhancement.

    This model can be used in two modes:

    1. Baseline U-Net:
       input_channels = 3

    2. DASP-Net:
       input_channels = 7
       RGB + illumination + edge + frequency + noise
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.inc = ConvBlock(input_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)

        self.up1 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)

        # Output image should be in range [0, 1]
        x = torch.sigmoid(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_baseline_unet() -> UNet:
    """
    Baseline model:
    input = RGB image
    channels = 3
    """
    return UNet(input_channels=3, output_channels=3, base_channels=32)


def build_dasp_net() -> UNet:
    """
    Proposed model:
    input = RGB + 4 prompt maps
    channels = 7
    """
    return UNet(input_channels=7, output_channels=3, base_channels=32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_model = build_baseline_unet().to(device)
    dasp_model = build_dasp_net().to(device)

    rgb_input = torch.randn(1, 3, 256, 256).to(device)
    dasp_input = torch.randn(1, 7, 256, 256).to(device)

    baseline_output = baseline_model(rgb_input)
    dasp_output = dasp_model(dasp_input)

    print("Device:", device)

    print("\nBaseline U-Net")
    print("Input shape:", rgb_input.shape)
    print("Output shape:", baseline_output.shape)
    print("Parameters:", count_parameters(baseline_model))

    print("\nDASP-Net")
    print("Input shape:", dasp_input.shape)
    print("Output shape:", dasp_output.shape)
    print("Parameters:", count_parameters(dasp_model))


if __name__ == "__main__":
    main()