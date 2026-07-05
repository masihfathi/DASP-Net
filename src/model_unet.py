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
            ConvBlock(in_channels, out_channels),
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

    Modes:
    1. Baseline U-Net:
       input_channels = 3

    2. Raw DASP-Net:
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
        x = torch.sigmoid(x)

        return x


class PromptGate(nn.Module):
    """
    Residual identity prompt gate.

    Previous gate version scaled RGB features directly in range [0.5, 1.5].
    That could change RGB features too strongly from the beginning of training.

    This version starts close to identity:

        output = feature * (1 + gamma * modulation)

    gamma is initialized to zero, so initially:

        output ≈ feature

    During training, gamma learns how much prompt maps should affect RGB features.
    """

    def __init__(self, prompt_channels: int, feature_channels: int):
        super().__init__()

        self.modulation = nn.Sequential(
            nn.Conv2d(prompt_channels, feature_channels, kernel_size=1),
            nn.Tanh(),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature, prompt_feature):
        modulation = self.modulation(prompt_feature)

        scale = 1.0 + self.gamma * modulation

        return feature * scale


class PromptGatedDASPNet(nn.Module):
    """
    Prompt-Gated DASP-Net v2.

    Input:
        7 channels:
        RGB + illumination + edge + frequency + noise

    RGB branch:
        extracts main visual features.

    Prompt branch:
        extracts lightweight guidance features from illumination, edge,
        frequency, and noise maps.

    Residual prompt gates:
        guide RGB features at multiple encoder levels while starting close
        to identity mapping.
    """

    def __init__(
        self,
        output_channels: int = 3,
        base_channels: int = 32,
        prompt_base_channels: int = 8,
    ):
        super().__init__()

        # RGB encoder
        self.rgb_inc = ConvBlock(3, base_channels)
        self.rgb_down1 = DownBlock(base_channels, base_channels * 2)
        self.rgb_down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.rgb_down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.rgb_down4 = DownBlock(base_channels * 8, base_channels * 16)

        # Lightweight prompt encoder
        self.prompt_inc = ConvBlock(4, prompt_base_channels)
        self.prompt_down1 = DownBlock(prompt_base_channels, prompt_base_channels * 2)
        self.prompt_down2 = DownBlock(prompt_base_channels * 2, prompt_base_channels * 4)
        self.prompt_down3 = DownBlock(prompt_base_channels * 4, prompt_base_channels * 8)
        self.prompt_down4 = DownBlock(prompt_base_channels * 8, prompt_base_channels * 16)

        # Residual prompt gates for RGB features
        self.gate1 = PromptGate(prompt_base_channels, base_channels)
        self.gate2 = PromptGate(prompt_base_channels * 2, base_channels * 2)
        self.gate3 = PromptGate(prompt_base_channels * 4, base_channels * 4)
        self.gate4 = PromptGate(prompt_base_channels * 8, base_channels * 8)
        self.gate5 = PromptGate(prompt_base_channels * 16, base_channels * 16)

        # Decoder
        self.up1 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x):
        if x.size(1) != 7:
            raise ValueError(f"PG-DASP-Net expects 7 input channels, but got {x.size(1)}.")

        rgb = x[:, 0:3, :, :]
        prompts = x[:, 3:7, :, :]

        # RGB encoder
        x1 = self.rgb_inc(rgb)
        x2 = self.rgb_down1(x1)
        x3 = self.rgb_down2(x2)
        x4 = self.rgb_down3(x3)
        x5 = self.rgb_down4(x4)

        # Prompt encoder
        p1 = self.prompt_inc(prompts)
        p2 = self.prompt_down1(p1)
        p3 = self.prompt_down2(p2)
        p4 = self.prompt_down3(p3)
        p5 = self.prompt_down4(p4)

        # Apply residual prompt gates
        x1 = self.gate1(x1, p1)
        x2 = self.gate2(x2, p2)
        x3 = self.gate3(x3, p3)
        x4 = self.gate4(x4, p4)
        x5 = self.gate5(x5, p5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)
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
    Raw DASP-Net:
    input = RGB + 4 prompt maps
    channels = 7
    """
    return UNet(input_channels=7, output_channels=3, base_channels=32)


def build_prompt_gated_dasp_net() -> PromptGatedDASPNet:
    """
    Prompt-Gated DASP-Net v2:
    input = RGB + 4 prompt maps
    channels = 7

    The prompt branch is intentionally lightweight.
    The residual gates start close to identity.
    """
    return PromptGatedDASPNet(
        output_channels=3,
        base_channels=32,
        prompt_base_channels=8,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_model = build_baseline_unet().to(device)
    dasp_model = build_dasp_net().to(device)
    pg_dasp_model = build_prompt_gated_dasp_net().to(device)

    rgb_input = torch.randn(1, 3, 256, 256).to(device)
    dasp_input = torch.randn(1, 7, 256, 256).to(device)

    baseline_output = baseline_model(rgb_input)
    dasp_output = dasp_model(dasp_input)
    pg_dasp_output = pg_dasp_model(dasp_input)

    print("Device:", device)

    print("\nBaseline U-Net")
    print("Input shape:", rgb_input.shape)
    print("Output shape:", baseline_output.shape)
    print("Parameters:", count_parameters(baseline_model))

    print("\nRaw DASP-Net")
    print("Input shape:", dasp_input.shape)
    print("Output shape:", dasp_output.shape)
    print("Parameters:", count_parameters(dasp_model))

    print("\nPG-DASP-Net v2")
    print("Input shape:", dasp_input.shape)
    print("Output shape:", pg_dasp_output.shape)
    print("Parameters:", count_parameters(pg_dasp_model))

    print("\nInitial gamma values in PG-DASP-Net v2")
    print("gate1 gamma:", pg_dasp_model.gate1.gamma.item())
    print("gate2 gamma:", pg_dasp_model.gate2.gamma.item())
    print("gate3 gamma:", pg_dasp_model.gate3.gamma.item())
    print("gate4 gamma:", pg_dasp_model.gate4.gamma.item())
    print("gate5 gamma:", pg_dasp_model.gate5.gamma.item())


if __name__ == "__main__":
    main()