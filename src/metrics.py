import torch
import torch.nn.functional as F


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Error.

    Inputs:
        pred:   B x C x H x W, range [0, 1]
        target: B x C x H x W, range [0, 1]

    Returns:
        scalar tensor
    """
    return torch.mean(torch.abs(pred - target))


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate PSNR in dB.

    Inputs:
        pred:   B x C x H x W, range [0, 1]
        target: B x C x H x W, range [0, 1]

    Returns:
        average PSNR over batch
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=(1, 2, 3))

    psnr = 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(mse + eps)

    return psnr.mean()


def create_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create Gaussian window for SSIM calculation.
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2

    gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = gaussian_1d[:, None] @ gaussian_1d[None, :]
    gaussian_2d = gaussian_2d / gaussian_2d.sum()

    window = gaussian_2d.expand(channels, 1, window_size, window_size).contiguous()

    return window


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate SSIM.

    Inputs:
        pred:   B x C x H x W, range [0, 1]
        target: B x C x H x W, range [0, 1]

    Returns:
        scalar tensor
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    _, channels, _, _ = pred.shape
    device = pred.device

    window = create_gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=device,
    )

    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2) + eps
    )

    return ssim_map.mean()


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Calculate all metrics used in this project.
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    metrics = {
        "mae": calculate_mae(pred, target).item(),
        "psnr": calculate_psnr(pred, target).item(),
        "ssim": calculate_ssim(pred, target).item(),
    }

    return metrics


def main():
    """
    Simple test for metrics.py
    """
    target = torch.rand(2, 3, 256, 256)

    # A slightly noisy prediction
    pred = target + 0.05 * torch.randn_like(target)
    pred = torch.clamp(pred, 0.0, 1.0)

    metrics = calculate_metrics(pred, target)

    print("Metrics test:")
    print("MAE:", metrics["mae"])
    print("PSNR:", metrics["psnr"])
    print("SSIM:", metrics["ssim"])


if __name__ == "__main__":
    main()