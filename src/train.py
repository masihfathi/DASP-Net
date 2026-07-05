import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import PairedLowLightDataset
from metrics import calculate_metrics, calculate_ssim
from model_unet import (
    build_baseline_unet,
    build_dasp_net,
    build_prompt_gated_dasp_net,
)


def get_device():
    """
    Select best available device:
    - CUDA for NVIDIA GPU
    - MPS for Apple Silicon GPU
    - CPU otherwise
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def gradient_loss(pred, target):
    """
    Simple gradient loss using image differences.
    Helps preserve edges and structures.
    """
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    loss_x = torch.mean(torch.abs(pred_dx - target_dx))
    loss_y = torch.mean(torch.abs(pred_dy - target_dy))

    return loss_x + loss_y


def total_loss(pred, target, lambda_ssim=0.2, lambda_grad=0.1):
    """
    Total loss used for training:

    L_total = L1 + lambda_ssim * (1 - SSIM) + lambda_grad * gradient_loss
    """
    l1 = nn.functional.l1_loss(pred, target)
    ssim_loss = 1.0 - calculate_ssim(pred, target)
    grad = gradient_loss(pred, target)

    loss = l1 + lambda_ssim * ssim_loss + lambda_grad * grad

    return loss, {
        "l1": l1.item(),
        "ssim_loss": ssim_loss.item(),
        "grad": grad.item(),
        "total": loss.item(),
    }


def save_sample_images(low_rgb, pred, target, output_dir, epoch, mode):
    """
    Save visual comparison:
    low image | enhanced image | target image
    """
    os.makedirs(output_dir, exist_ok=True)

    low_rgb = low_rgb.detach().cpu()
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    comparison = torch.cat([low_rgb, pred, target], dim=0)

    save_path = os.path.join(output_dir, f"{mode}_epoch_{epoch:03d}.png")
    save_image(comparison, save_path, nrow=low_rgb.size(0))


def get_model_input(mode, low_rgb, dasp_input):
    """
    Select correct model input based on training mode.

    baseline:
        uses RGB input only.

    dasp:
        uses 7-channel input:
        RGB + illumination + edge + frequency + noise.

    pgdasp:
        uses 7-channel input:
        RGB + illumination + edge + frequency + noise.
    """
    if mode == "baseline":
        return low_rgb

    if mode in ["dasp", "pgdasp"]:
        return dasp_input

    raise ValueError(f"Unknown mode: {mode}")


def build_model(mode):
    """
    Build model according to selected mode.
    """
    if mode == "baseline":
        return build_baseline_unet()

    if mode == "dasp":
        return build_dasp_net()

    if mode == "pgdasp":
        return build_prompt_gated_dasp_net()

    raise ValueError("mode must be one of: 'baseline', 'dasp', or 'pgdasp'")


def evaluate(model, loader, device, mode, output_dir, epoch):
    """
    Evaluate model on validation/test dataset.
    """
    model.eval()

    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_count = 0

    first_batch_saved = False

    with torch.no_grad():
        for batch in loader:
            low_rgb = batch["low_rgb"].to(device)
            dasp_input = batch["dasp_input"].to(device)
            target = batch["target"].to(device)

            model_input = get_model_input(
                mode=mode,
                low_rgb=low_rgb,
                dasp_input=dasp_input,
            )

            pred = model(model_input)

            metrics = calculate_metrics(pred, target)

            batch_size = low_rgb.size(0)

            total_mae += metrics["mae"] * batch_size
            total_psnr += metrics["psnr"] * batch_size
            total_ssim += metrics["ssim"] * batch_size
            total_count += batch_size

            if not first_batch_saved:
                save_sample_images(
                    low_rgb=low_rgb,
                    pred=pred,
                    target=target,
                    output_dir=output_dir,
                    epoch=epoch,
                    mode=mode,
                )
                first_batch_saved = True

    avg_metrics = {
        "mae": total_mae / total_count,
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
    }

    return avg_metrics


def train(args):
    device = get_device()
    print("Device:", device)
    print("Training mode:", args.mode)

    train_dataset = PairedLowLightDataset(
        low_dir=args.train_low_dir,
        high_dir=args.train_high_dir,
        image_size=(args.height, args.width),
        use_dasp_input=True,
    )

    val_dataset = PairedLowLightDataset(
        low_dir=args.val_low_dir,
        high_dir=args.val_high_dir,
        image_size=(args.height, args.width),
        use_dasp_input=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.mode)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    sample_dir = Path(args.output_dir) / "samples"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_ssim_loss = 0.0
        epoch_grad = 0.0
        total_count = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            low_rgb = batch["low_rgb"].to(device)
            dasp_input = batch["dasp_input"].to(device)
            target = batch["target"].to(device)

            model_input = get_model_input(
                mode=args.mode,
                low_rgb=low_rgb,
                dasp_input=dasp_input,
            )

            pred = model(model_input)

            loss, loss_items = total_loss(
                pred,
                target,
                lambda_ssim=args.lambda_ssim,
                lambda_grad=args.lambda_grad,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = low_rgb.size(0)

            epoch_loss += loss_items["total"] * batch_size
            epoch_l1 += loss_items["l1"] * batch_size
            epoch_ssim_loss += loss_items["ssim_loss"] * batch_size
            epoch_grad += loss_items["grad"] * batch_size
            total_count += batch_size

            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss_items['total']:.4f} "
                    f"L1: {loss_items['l1']:.4f} "
                    f"SSIM_Loss: {loss_items['ssim_loss']:.4f} "
                    f"Grad: {loss_items['grad']:.4f}"
                )

        avg_train_loss = epoch_loss / total_count
        avg_l1 = epoch_l1 / total_count
        avg_ssim_loss = epoch_ssim_loss / total_count
        avg_grad = epoch_grad / total_count

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mode=args.mode,
            output_dir=str(sample_dir),
            epoch=epoch,
        )

        print("=" * 70)
        print(f"Epoch {epoch}/{args.epochs} finished.")
        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            f"L1: {avg_l1:.4f} | "
            f"SSIM Loss: {avg_ssim_loss:.4f} | "
            f"Grad: {avg_grad:.4f}"
        )
        print(
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val PSNR: {val_metrics['psnr']:.4f} | "
            f"Val SSIM: {val_metrics['ssim']:.4f}"
        )
        print("=" * 70)

        last_checkpoint = checkpoint_dir / f"{args.mode}_last.pth"
        torch.save(
            {
                "epoch": epoch,
                "mode": args.mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            },
            last_checkpoint,
        )

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]

            best_checkpoint = checkpoint_dir / f"{args.mode}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "mode": args.mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_checkpoint,
            )

            print(f"New best model saved: {best_checkpoint}")


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net baseline, raw DASP-Net, or PG-DASP-Net on LOL Dataset."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "dasp", "pgdasp"],
        help="Training mode: baseline, dasp, or pgdasp.",
    )

    parser.add_argument("--train-low-dir", type=str, required=True)
    parser.add_argument("--train-high-dir", type=str, required=True)
    parser.add_argument("--val-low-dir", type=str, required=True)
    parser.add_argument("--val-high-dir", type=str, required=True)

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--lambda-grad", type=float, default=0.1)

    parser.add_argument("--output-dir", type=str, default="results/training")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()