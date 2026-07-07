import argparse
import csv
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Summarize external training best checkpoints.")
    parser.add_argument("--root", default="results", help="Root folder to search.")
    parser.add_argument("--output-csv", default="results/external_training_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    checkpoints = sorted(root.rglob("*_best.pth"))

    rows = []
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue

        if "dataset_name" not in ckpt:
            continue

        metrics = ckpt.get("best_metrics", {})
        rows.append({
            "checkpoint": str(ckpt_path),
            "dataset": ckpt.get("dataset_name", "-"),
            "epoch": ckpt.get("epoch", "-"),
            "mode": ckpt.get("mode", "-"),
            "prompt_mode": ckpt.get("prompt_mode", "-"),
            "mae": metrics.get("mae", float("nan")),
            "psnr": metrics.get("psnr", float("nan")),
            "ssim": metrics.get("ssim", float("nan")),
        })

    print("| Dataset | Mode | Prompt | Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ | Checkpoint |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        print(
            f"| {r['dataset']} | {r['mode']} | {r['prompt_mode']} | {r['epoch']} | "
            f"{r['mae']:.4f} | {r['psnr']:.4f} | {r['ssim']:.4f} | `{r['checkpoint']}` |"
        )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "dataset", "epoch", "mode", "prompt_mode", "mae", "psnr", "ssim"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
