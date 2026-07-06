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


class StrengthControlledPromptGate(nn.Module):
    """
    Residual prompt gate controlled by prompt strength.

    Standard residual gate:
        F' = F * (1 + gamma * M)

    This v3 gate:
        F' = F * (1 + gamma * S * M)

    where:
        S = 1 - alpha_no_prompt

    If the selector assigns high weight to the no-prompt option,
    S becomes small and prompt influence is actually suppressed after
    the prompt encoder and BatchNorm layers.
    """

    def __init__(self, prompt_channels: int, feature_channels: int):
        super().__init__()

        self.modulation = nn.Sequential(
            nn.Conv2d(prompt_channels, feature_channels, kernel_size=1),
            nn.Tanh(),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature, prompt_feature, prompt_strength):
        """
        feature:
            B x C x H x W

        prompt_feature:
            B x Cp x H x W

        prompt_strength:
            B x 1
        """
        modulation = self.modulation(prompt_feature)

        strength = prompt_strength.view(prompt_strength.size(0), 1, 1, 1)
        scale = 1.0 + self.gamma * strength * modulation

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


class AdaptivePromptSelection(nn.Module):
    """
    Adaptive prompt selection with an explicit no-prompt option.

    Input prompt channel order:
        0: illumination
        1: edge
        2: frequency
        3: noise

    The selector learns five image-dependent weights:

        alpha_0: no-prompt option
        alpha_1: illumination
        alpha_2: edge
        alpha_3: frequency
        alpha_4: noise

    The no-prompt weight does not correspond to a map. It controls how much
    prompt influence should be suppressed.

    Prompt strength:
        S = 1 - alpha_0

    Weighted prompts:
        P_weighted = [alpha_1 P_illum,
                      alpha_2 P_edge,
                      alpha_3 P_freq,
                      alpha_4 P_noise]
    """

    def __init__(
        self,
        prompt_channels: int = 4,
        hidden_dim: int = 16,
        no_prompt_init_bias: float = 2.0,
    ):
        super().__init__()

        self.prompt_channels = prompt_channels

        self.fc1 = nn.Linear(prompt_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, prompt_channels + 1)

        # Stable initialization:
        # Start near no-prompt behavior.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[0] = no_prompt_init_bias

    def forward(self, prompts):
        """
        prompts shape:
            B x 4 x H x W

        returns:
            weighted_prompts: B x 4 x H x W
            selection_weights: B x 5
                [no_prompt, illumination, edge, frequency, noise]
            prompt_strength: B x 1
                1 - no_prompt
        """
        if prompts.size(1) != self.prompt_channels:
            raise ValueError(
                f"AdaptivePromptSelection expects {self.prompt_channels} prompt channels, "
                f"but got {prompts.size(1)}."
            )

        batch_size = prompts.size(0)

        # Global mean statistic for each prompt map
        stats = torch.mean(prompts, dim=(2, 3))  # B x 4

        logits = self.fc2(self.relu(self.fc1(stats)))  # B x 5
        selection_weights = torch.softmax(logits, dim=1)

        no_prompt_weight = selection_weights[:, 0:1]   # B x 1
        prompt_strength = 1.0 - no_prompt_weight       # B x 1

        # alpha_1..alpha_4
        prompt_weights = selection_weights[:, 1:]      # B x 4

        weighted_prompts = prompts * prompt_weights.view(batch_size, 4, 1, 1)

        return weighted_prompts, selection_weights, prompt_strength


class AdaptivePromptGatedDASPNet(nn.Module):
    """
    APG-DASP-Net v3:
    Adaptive Prompt-Gated DASP-Net with no-prompt controlled residual gating.

    Input:
        7 channels:
        RGB + illumination + edge + frequency + noise

    Main idea:
        The network learns image-dependent weights for four prompt maps plus
        an explicit no-prompt option.

        In v2, prompt maps were weakened before the prompt encoder, but BatchNorm
        could partially cancel this weakening.

        In v3, the no-prompt option also directly controls the residual gates:

            F' = F * (1 + gamma * S * M)

        where:
            S = 1 - alpha_no_prompt

        Therefore, if prompts are not useful, the model can suppress prompt
        influence after the prompt encoder as well.
    """

    def __init__(
        self,
        output_channels: int = 3,
        base_channels: int = 32,
        prompt_base_channels: int = 8,
        hidden_dim: int = 16,
        no_prompt_init_bias: float = 2.0,
    ):
        super().__init__()

        self.prompt_selection = AdaptivePromptSelection(
            prompt_channels=4,
            hidden_dim=hidden_dim,
            no_prompt_init_bias=no_prompt_init_bias,
        )

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

        # Strength-controlled residual prompt gates
        self.gate1 = StrengthControlledPromptGate(prompt_base_channels, base_channels)
        self.gate2 = StrengthControlledPromptGate(prompt_base_channels * 2, base_channels * 2)
        self.gate3 = StrengthControlledPromptGate(prompt_base_channels * 4, base_channels * 4)
        self.gate4 = StrengthControlledPromptGate(prompt_base_channels * 8, base_channels * 8)
        self.gate5 = StrengthControlledPromptGate(prompt_base_channels * 16, base_channels * 16)

        # Decoder
        self.up1 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

        # Optional storage for analysis
        self.last_selection_weights = None  # B x 5
        self.last_prompt_weights = None     # B x 4, for compatibility with train.py
        self.last_prompt_strength = None    # B x 1

    def forward(self, x):
        if x.size(1) != 7:
            raise ValueError(
                f"Adaptive PG-DASP-Net expects 7 input channels, but got {x.size(1)}."
            )

        rgb = x[:, 0:3, :, :]
        prompts = x[:, 3:7, :, :]

        weighted_prompts, selection_weights, prompt_strength = self.prompt_selection(prompts)

        self.last_selection_weights = selection_weights.detach()
        self.last_prompt_weights = selection_weights[:, 1:].detach()
        self.last_prompt_strength = prompt_strength.detach()

        # RGB encoder
        x1 = self.rgb_inc(rgb)
        x2 = self.rgb_down1(x1)
        x3 = self.rgb_down2(x2)
        x4 = self.rgb_down3(x3)
        x5 = self.rgb_down4(x4)

        # Prompt encoder with selected weighted prompts
        p1 = self.prompt_inc(weighted_prompts)
        p2 = self.prompt_down1(p1)
        p3 = self.prompt_down2(p2)
        p4 = self.prompt_down3(p3)
        p5 = self.prompt_down4(p4)

        # Apply strength-controlled residual gates
        x1 = self.gate1(x1, p1, prompt_strength)
        x2 = self.gate2(x2, p2, prompt_strength)
        x3 = self.gate3(x3, p3, prompt_strength)
        x4 = self.gate4(x4, p4, prompt_strength)
        x5 = self.gate5(x5, p5, prompt_strength)

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


def build_adaptive_prompt_gated_dasp_net() -> AdaptivePromptGatedDASPNet:
    """
    APG-DASP-Net v3:
    input = RGB + 4 prompt maps
    channels = 7

    It learns image-dependent weights for four prompt maps plus an explicit
    no-prompt option, and directly controls the residual gating strength.
    """
    return AdaptivePromptGatedDASPNet(
        output_channels=3,
        base_channels=32,
        prompt_base_channels=8,
        hidden_dim=16,
        no_prompt_init_bias=2.0,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_model = build_baseline_unet().to(device)
    dasp_model = build_dasp_net().to(device)
    pg_dasp_model = build_prompt_gated_dasp_net().to(device)
    adaptive_pg_dasp_model = build_adaptive_prompt_gated_dasp_net().to(device)

    rgb_input = torch.randn(1, 3, 256, 256).to(device)
    dasp_input = torch.randn(1, 7, 256, 256).to(device)

    baseline_output = baseline_model(rgb_input)
    dasp_output = dasp_model(dasp_input)
    pg_dasp_output = pg_dasp_model(dasp_input)
    adaptive_pg_dasp_output = adaptive_pg_dasp_model(dasp_input)

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

    print("\nAPG-DASP-Net v3 with No-Prompt Controlled Gating")
    print("Input shape:", dasp_input.shape)
    print("Output shape:", adaptive_pg_dasp_output.shape)
    print("Parameters:", count_parameters(adaptive_pg_dasp_model))

    print("\nInitial gamma values in PG-DASP-Net v2")
    print("gate1 gamma:", pg_dasp_model.gate1.gamma.item())
    print("gate2 gamma:", pg_dasp_model.gate2.gamma.item())
    print("gate3 gamma:", pg_dasp_model.gate3.gamma.item())
    print("gate4 gamma:", pg_dasp_model.gate4.gamma.item())
    print("gate5 gamma:", pg_dasp_model.gate5.gamma.item())

    print("\nInitial gamma values in APG-DASP-Net v3")
    print("gate1 gamma:", adaptive_pg_dasp_model.gate1.gamma.item())
    print("gate2 gamma:", adaptive_pg_dasp_model.gate2.gamma.item())
    print("gate3 gamma:", adaptive_pg_dasp_model.gate3.gamma.item())
    print("gate4 gamma:", adaptive_pg_dasp_model.gate4.gamma.item())
    print("gate5 gamma:", adaptive_pg_dasp_model.gate5.gamma.item())

    print("\nAdaptive selection weights for the test input")
    selection = adaptive_pg_dasp_model.last_selection_weights
    strength = adaptive_pg_dasp_model.last_prompt_strength

    if selection is not None:
        print("weights shape:", selection.shape)
        print(
            "no_prompt, illumination, edge, frequency, noise:",
            selection[0].detach().cpu().tolist(),
        )

    if strength is not None:
        print("prompt_strength:", strength[0].item())


if __name__ == "__main__":
    main()
