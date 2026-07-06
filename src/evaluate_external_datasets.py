import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Allow local imports when running as:
# python3 src/evaluate_external_datasets.py
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
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def read_rgb(path):
    return Image.open(path).convert("RGB")


def resize_image(img, size):
    if size is None:
        return img
    h, w = size
    return img.resize((w, h), Image.BICUBIC)


def pil_to_numpy_rgb(img):
    return np.asarray(img).astype(np.float32) / 255.0


def numpy_hwc_to_tensor_chw(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] in [1, 3, 4, 7]:
        return torch.from_numpy(arr).permute(2, 0, 1).float()

    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4, 7]:
        return torch.from_numpy(arr).float()

    raise ValueError(f"Unsupported image/prompt array shape: {arr.shape}")


def build_dasp_tensor_from_numpy(low_np):
    dasp_np = build_dasp_input(low_np)
    return numpy_hwc_to_tensor_chw(dasp_np)


class ExternalDataset(Dataset):
    def __init__(self, low_dir, high_dir=None, image_size=(256, 256)):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir) if high_dir else None
        self.image_size = image_size

        low_files = list_images(self.low_dir)
        if not low_files:
            raise FileNotFoundError(f"No images found in low_dir: {self.low_dir}")

        if self.high_dir is None:
            self.items = [(p, None) for p in low_files]
        else:
            high_files = list_images(self.high_dir)
            high_by_stem = {p.stem: p for p in high_files}

            items = []
            missing = []
            for low_path in low_files:
                high_path = high_by_stem.get(low_path.stem)
                if high_path is None:
                    missing.append(low_path.name)
                else:
                    items.append((low_path, high_path))

            if not items:
                raise FileNotFoundError(
                    f"No paired images matched by filename stem between {self.low_dir} and {self.high_dir}"
                )

            if missing:
                print(f"[warning] {len(missing)} low images had no matching high image and were skipped.")

            self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        low_path, high_path = self.items[idx]

        low_img = read_rgb(low_path)
        original_size = low_img.size

        low_img = resize_image(low_img, self.image_size)
        low_np = pil_to_numpy_rgb(low_img)

        low_rgb = numpy_hwc_to_tensor_chw(low_np)
        dasp_input = build_dasp_tensor_from_numpy(low_np)

        sample = {
            "name": low_path.stem,
            "low_path": str(low_path),
            "low_rgb": low_rgb,
            "dasp_input": dasp_input,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

        if high_path is not None:
            high_img = read_rgb(high_path)
            high_img = resize_image(high_img, self.image_size)
            high_np = pil_to_numpy_rgb(high_img)
            sample["target"] = numpy_hwc_to_tensor_chw(high_np)
            sample["high_path"] = str(high_path)

        return sample


def apply_prompt_ablation(dasp_input, prompt_mode):
    if prompt_mode in [None, "", "-", "full"]:
        return dasp_input

    output = dasp_input.clone()
    output[:, 3:7, :, :] = 0.0

    if prompt_mode == "none":
        return output

    if prompt_mode == "illumination":
        output[:, 3:4, :, :] = dasp_input[:, 3:4, :, :]
    elif prompt_mode == "edge":
        output[:, 4:5, :, :] = dasp_input[:, 4:5, :, :]
    elif prompt_mode == "frequency":
        output[:, 5:6, :, :] = dasp_input[:, 5:6, :, :]
    elif prompt_mode == "noise":
        output[:, 6:7, :, :] = dasp_input[:, 6:7, :, :]
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    return output


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


def get_input(mode, low_rgb, dasp_input):
    if mode == "baseline":
        return low_rgb
    if mode in ["dasp", "pgdasp", "apgdasp"]:
        return dasp_input
    raise ValueError(f"Unknown mode: {mode}")


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = tensor.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def try_load_lpips(device):
    try:
        import lpips
        model = lpips.LPIPS(net="alex")
        model = model.to(device)
        model.eval()
        print("[info] LPIPS enabled.")
        return model
    except Exception as exc:
        print(f"[warning] LPIPS disabled: {exc}")
        return None


def lpips_score(lpips_model, pred, target):
    if lpips_model is None:
        return math.nan

    pred_lp = pred * 2.0 - 1.0
    target_lp = target * 2.0 - 1.0

    with torch.no_grad():
        value = lpips_model(pred_lp, target_lp)

    return float(value.mean().item())


def parse_checkpoint_item(item):
    if ":" not in item:
        raise ValueError("Checkpoint must be in name:path format.")
    name, path = item.split(":", 1)
    return name.strip(), Path(path.strip())


def load_checkpoint_model(checkpoint_path, device, strict=False):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    mode = ckpt.get("mode")
    prompt_mode = ckpt.get("prompt_mode", "full")
    epoch = ckpt.get("epoch", -1)

    if mode is None and "args" in ckpt:
        mode = ckpt["args"].get("mode")
        prompt_mode = ckpt["args"].get("prompt_mode", prompt_mode)

    model = build_model(mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.eval()

    return model, mode, prompt_mode, epoch


def evaluate_model(
    dataset_name,
    model_name,
    checkpoint_path,
    loader,
    paired,
    device,
    lpips_model,
    save_outputs_dir=None,
    strict_load=False,
):
    model, mode, prompt_mode, epoch = load_checkpoint_model(checkpoint_path, device, strict=strict_load)

    totals = {
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }

    count = 0
    lpips_count = 0

    if save_outputs_dir:
        save_outputs_dir = Path(save_outputs_dir) / dataset_name / model_name.replace("/", "_").replace(" ", "_")
        save_outputs_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            low_rgb = batch["low_rgb"].to(device)
            dasp_input = batch["dasp_input"].to(device)

            if mode in ["dasp", "pgdasp", "apgdasp"]:
                dasp_input = apply_prompt_ablation(dasp_input, prompt_mode)

            model_input = get_input(mode, low_rgb, dasp_input)
            pred = model(model_input).clamp(0.0, 1.0)

            batch_size = pred.size(0)

            if paired:
                target = batch["target"].to(device)
                metrics = calculate_metrics(pred, target)
                totals["mae"] += metrics["mae"] * batch_size
                totals["psnr"] += metrics["psnr"] * batch_size
                totals["ssim"] += metrics["ssim"] * batch_size

                lp = lpips_score(lpips_model, pred, target)
                if not math.isnan(lp):
                    totals["lpips"] += lp * batch_size
                    lpips_count += batch_size

            if save_outputs_dir:
                for i in range(batch_size):
                    name = batch["name"][i]
                    out_img = tensor_to_image(pred[i])
                    out_img.save(save_outputs_dir / f"{name}.png")

            count += batch_size

    return {
        "dataset": dataset_name,
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "mode": mode,
        "prompt_mode": prompt_mode,
        "num_images": count,
        "paired": paired,
        "mae": totals["mae"] / count if paired else math.nan,
        "psnr": totals["psnr"] / count if paired else math.nan,
        "ssim": totals["ssim"] / count if paired else math.nan,
        "lpips": totals["lpips"] / lpips_count if lpips_count > 0 else math.nan,
        "niqe": math.nan,
    }


def save_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "checkpoint",
        "epoch",
        "mode",
        "prompt_mode",
        "num_images",
        "paired",
        "mae",
        "psnr",
        "ssim",
        "lpips",
        "niqe",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows):
    print("| Dataset | Model | Images | Paired | MAE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")

    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "-"
            return f"{v:.4f}"
        return str(v)

    for r in rows:
        print(
            f"| {r['dataset']} | {r['model']} | {r['num_images']} | {r['paired']} | "
            f"{fmt(r['mae'])} | {fmt(r['psnr'])} | {fmt(r['ssim'])} | {fmt(r['lpips'])} | {fmt(r['niqe'])} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DASP-Net checkpoints on paired or unpaired external datasets. NIQE is skipped for speed and MPS compatibility."
    )

    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--low-dir", type=str, required=True)
    parser.add_argument("--high-dir", type=str, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint in name:path format. Can be repeated.",
    )

    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--save-outputs-dir", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict-load", action="store_true")

    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)
    print("Device:", device)
    print("[info] NIQE is skipped in this fast evaluator.")

    paired = args.high_dir is not None

    dataset = ExternalDataset(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        image_size=(args.height, args.width),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    lpips_model = try_load_lpips(device) if paired else None
    checkpoints = [parse_checkpoint_item(item) for item in args.checkpoint]

    rows = []
    for model_name, ckpt_path in checkpoints:
        print(f"\nEvaluating {model_name} on {args.dataset_name}")
        row = evaluate_model(
            dataset_name=args.dataset_name,
            model_name=model_name,
            checkpoint_path=ckpt_path,
            loader=loader,
            paired=paired,
            device=device,
            lpips_model=lpips_model,
            save_outputs_dir=args.save_outputs_dir,
            strict_load=args.strict_load,
        )
        rows.append(row)

    print("\nFinal table:")
    print_markdown(rows)

    save_csv(rows, args.output_csv)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
