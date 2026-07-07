import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from prompts import build_dasp_input
from metrics import calculate_metrics
from model_unet import (
    build_baseline_unet,
    build_dasp_net,
    build_prompt_gated_dasp_net,
    build_adaptive_prompt_gated_dasp_net,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def read_rgb(path):
    return Image.open(path).convert("RGB")


def resize_image(img, height, width):
    return img.resize((width, height), Image.BICUBIC)


def pil_to_numpy_rgb(img):
    return np.asarray(img).astype(np.float32) / 255.0


def array_to_chw_tensor(arr):
    """
    Convert image/prompt arrays to C x H x W tensor.

    RGB images from PIL are H x W x 3.
    build_dasp_input() in this project may return either:
      - H x W x 7, or
      - 7 x H x W

    The previous version always permuted HWC -> CHW, so if build_dasp_input()
    returned 7 x H x W, it became W x 7 x H and caused:
      expected 7 channels, but got 256 channels.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape: {arr.shape}")

    # Already CHW.
    if arr.shape[0] in (1, 3, 4, 7) and arr.shape[1] > 16 and arr.shape[2] > 16:
        return torch.from_numpy(arr).float()

    # HWC.
    if arr.shape[-1] in (1, 3, 4, 7) and arr.shape[0] > 16 and arr.shape[1] > 16:
        return torch.from_numpy(arr).permute(2, 0, 1).float()

    raise ValueError(f"Unsupported array shape for CHW conversion: {arr.shape}")


class PairedExternalDataset(Dataset):
    def __init__(self, low_dir, high_dir, height=256, width=256):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.height = height
        self.width = width

        low_files = list_images(self.low_dir)
        high_files = list_images(self.high_dir)
        high_by_stem = {p.stem: p for p in high_files}

        self.items = []
        missing = 0
        for low_path in low_files:
            high_path = high_by_stem.get(low_path.stem)
            if high_path is None:
                missing += 1
            else:
                self.items.append((low_path, high_path))

        if not self.items:
            raise FileNotFoundError(
                f"No paired images matched by filename stem between {self.low_dir} and {self.high_dir}"
            )

        if missing:
            print(f"[warning] {missing} low images were skipped because no matching high image was found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        low_path, high_path = self.items[idx]

        low_img = resize_image(read_rgb(low_path), self.height, self.width)
        high_img = resize_image(read_rgb(high_path), self.height, self.width)

        low_np = pil_to_numpy_rgb(low_img)
        high_np = pil_to_numpy_rgb(high_img)

        low_rgb = array_to_chw_tensor(low_np)
        target = array_to_chw_tensor(high_np)

        dasp_np = build_dasp_input(low_np)
        dasp_input = array_to_chw_tensor(dasp_np)

        return {
            "name": low_path.stem,
            "low_rgb": low_rgb,
            "dasp_input": dasp_input,
            "target": target,
        }


def apply_prompt_ablation(dasp_input, prompt_mode):
    if prompt_mode in [None, "", "-", "full"]:
        return dasp_input

    out = dasp_input.clone()
    out[:, 3:7, :, :] = 0.0

    if prompt_mode == "none":
        return out

    if prompt_mode == "illumination":
        out[:, 3:4, :, :] = dasp_input[:, 3:4, :, :]
    elif prompt_mode == "edge":
        out[:, 4:5, :, :] = dasp_input[:, 4:5, :, :]
    elif prompt_mode == "frequency":
        out[:, 5:6, :, :] = dasp_input[:, 5:6, :, :]
    elif prompt_mode == "noise":
        out[:, 6:7, :, :] = dasp_input[:, 6:7, :, :]
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    return out


def build_model(mode):
    if mode == "baseline":
        return build_baseline_unet()
    if mode == "dasp":
        return build_dasp_net()
    if mode == "pgdasp":
        return build_prompt_gated_dasp_net()
    if mode == "apgdasp":
        return build_adaptive_prompt_gated_dasp_net()
    raise ValueError(f"Unknown mode: {mode}")


def get_model_input(mode, low_rgb, dasp_input):
    if mode == "baseline":
        return low_rgb
    return dasp_input


def differentiable_ssim(x, y, window_size=11):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)

    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )

    return ssim_map.mean()


def gradient_loss(pred, target):
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def total_loss(pred, target, lambda_ssim=0.2, lambda_grad=0.1):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1.0 - differentiable_ssim(pred, target)
    grad = gradient_loss(pred, target)
    loss = l1 + lambda_ssim * ssim_loss + lambda_grad * grad
    return loss, {
        "l1": float(l1.detach().cpu()),
        "ssim_loss": float(ssim_loss.detach().cpu()),
        "grad": float(grad.detach().cpu()),
    }


def evaluate(model, loader, mode, prompt_mode, device):
    model.eval()

    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            low_rgb = batch["low_rgb"].to(device)
            dasp_input = batch["dasp_input"].to(device)
            target = batch["target"].to(device)

            if mode in ["dasp", "pgdasp", "apgdasp"]:
                dasp_input = apply_prompt_ablation(dasp_input, prompt_mode)

            inp = get_model_input(mode, low_rgb, dasp_input)
            pred = model(inp).clamp(0, 1)

            metrics = calculate_metrics(pred, target)
            bs = pred.size(0)

            total_mae += metrics["mae"] * bs
            total_psnr += metrics["psnr"] * bs
            total_ssim += metrics["ssim"] * bs
            n += bs

    return {
        "mae": total_mae / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
    }


def save_history_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_l1",
        "train_ssim_loss",
        "train_grad",
        "val_mae",
        "val_psnr",
        "val_ssim",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description="Train DASP-Net variants on any paired low-light dataset.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--train-low-dir", required=True)
    parser.add_argument("--train-high-dir", required=True)
    parser.add_argument("--val-low-dir", required=True)
    parser.add_argument("--val-high-dir", required=True)

    parser.add_argument("--mode", choices=["baseline", "dasp", "pgdasp", "apgdasp"], required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=["full", "none", "illumination", "edge", "frequency", "noise"],
        default="full",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--lambda-grad", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Mode: {args.mode}")
    print(f"Prompt mode: {args.prompt_mode}")

    train_dataset = PairedExternalDataset(
        args.train_low_dir,
        args.train_high_dir,
        height=args.height,
        width=args.width,
    )
    val_dataset = PairedExternalDataset(
        args.val_low_dir,
        args.val_high_dir,
        height=args.height,
        width=args.width,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    model = build_model(args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0
    best_row = None
    history = []

    ckpt_name = f"{args.mode}_{args.prompt_mode}_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0
        running_grad = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in progress:
            low_rgb = batch["low_rgb"].to(device)
            dasp_input = batch["dasp_input"].to(device)
            target = batch["target"].to(device)

            if args.mode in ["dasp", "pgdasp", "apgdasp"]:
                dasp_input = apply_prompt_ablation(dasp_input, args.prompt_mode)

            inp = get_model_input(args.mode, low_rgb, dasp_input)
            pred = model(inp).clamp(0, 1)

            loss, parts = total_loss(
                pred,
                target,
                lambda_ssim=args.lambda_ssim,
                lambda_grad=args.lambda_grad,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = pred.size(0)
            running_loss += float(loss.detach().cpu()) * bs
            running_l1 += parts["l1"] * bs
            running_ssim += parts["ssim_loss"] * bs
            running_grad += parts["grad"] * bs
            seen += bs

            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        val_metrics = evaluate(model, val_loader, args.mode, args.prompt_mode, device)

        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_l1": running_l1 / seen,
            "train_ssim_loss": running_ssim / seen,
            "train_grad": running_grad / seen,
            "val_mae": val_metrics["mae"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {row['train_loss']:.4f} | "
            f"L1: {row['train_l1']:.4f} | "
            f"SSIM Loss: {row['train_ssim_loss']:.4f} | "
            f"Grad: {row['train_grad']:.4f} | "
            f"Val MAE: {row['val_mae']:.4f} | "
            f"Val PSNR: {row['val_psnr']:.4f} | "
            f"Val SSIM: {row['val_ssim']:.4f}"
        )

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            best_row = row

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "mode": args.mode,
                "prompt_mode": args.prompt_mode,
                "dataset_name": args.dataset_name,
                "best_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / ckpt_name)
            print(f"New best model saved: {ckpt_dir / ckpt_name}")

        save_history_csv(history, output_dir / "history.csv")

    print("\nBest validation result:")
    print("| Dataset | Mode | Prompt | Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |")
    print("|---|---|---|---:|---:|---:|---:|")
    print(
        f"| {args.dataset_name} | {args.mode} | {args.prompt_mode} | {best_row['epoch']} | "
        f"{best_row['val_mae']:.4f} | {best_row['val_psnr']:.4f} | {best_row['val_ssim']:.4f} |"
    )


if __name__ == "__main__":
    main()
