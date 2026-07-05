import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    args = parser.parse_args()

    root = Path(args.root)

    checkpoint_files = sorted(root.rglob("*_best.pth"))

    if len(checkpoint_files) == 0:
        print("No best checkpoints found.")
        return

    print("| Checkpoint | Epoch | Mode | Prompt | MAE | PSNR | SSIM |")
    print("|---|---:|---|---|---:|---:|---:|")

    for ckpt_path in checkpoint_files:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        epoch = ckpt.get("epoch", "-")
        mode = ckpt.get("mode", "-")
        prompt_mode = ckpt.get("prompt_mode", "-")
        metrics = ckpt.get("val_metrics", {})

        mae = metrics.get("mae", None)
        psnr = metrics.get("psnr", None)
        ssim = metrics.get("ssim", None)

        mae_str = f"{mae:.4f}" if mae is not None else "-"
        psnr_str = f"{psnr:.4f}" if psnr is not None else "-"
        ssim_str = f"{ssim:.4f}" if ssim is not None else "-"

        print(
            f"| {ckpt_path} | {epoch} | {mode} | {prompt_mode} | "
            f"{mae_str} | {psnr_str} | {ssim_str} |"
        )


if __name__ == "__main__":
    main()