import argparse
import csv
import math
import sys
from pathlib import Path

# Make local src imports work when the script is executed as:
# python3 src/evaluate_paper_metrics.py
sys.path.append(str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from dataset import PairedLowLightDataset
from metrics import calculate_metrics
from model_unet import (
    build_baseline_unet,
    build_dasp_net,
    build_prompt_gated_dasp_net,
    build_adaptive_prompt_gated_dasp_net,
)


DEFAULT_CHECKPOINTS = [
    ("U-Net", "results/baseline_20epoch/checkpoints/baseline_best.pth"),
    ("Raw DASP-Net", "results/dasp_20epoch/checkpoints/dasp_best.pth"),
    ("PG-DASP full", "results/pgdasp_v2_20epoch/checkpoints/pgdasp_best.pth"),
    ("PG-DASP no prompt", "results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth"),
    ("APG-DASP v3", "results/apgdasp_v3_20epoch/checkpoints/apgdasp_best.pth"),
]


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def apply_prompt_ablation(dasp_input, prompt_mode):
    """
    Channel order:
        0 R
        1 G
        2 B
        3 illumination
        4 edge
        5 frequency
        6 noise
    """
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


def try_load_lpips(device):
    """
    LPIPS is a full-reference perceptual metric.
    It requires the lpips package.

    Install:
        python3 -m pip install lpips
    """
    try:
        import lpips

        model = lpips.LPIPS(net="alex")
        model = model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"[warning] LPIPS disabled: {exc}")
        return None


def try_load_niqe(device):
    """
    NIQE is a no-reference image quality metric.

    The piq package version installed on some systems does not include NIQE.
    Therefore this function first tries pyiqa, which supports NIQE more reliably.

    Install:
        python3 -m pip install pyiqa

    If pyiqa is unavailable, NIQE will be skipped instead of crashing.
    """
    try:
        import pyiqa

        metric = pyiqa.create_metric("niqe", device=device)
        metric.eval()
        print("[info] NIQE enabled using pyiqa.")
        return ("pyiqa", metric)
    except Exception as exc:
        print(f"[warning] pyiqa NIQE unavailable: {exc}")

    try:
        import piq

        if hasattr(piq, "niqe"):
            print("[info] NIQE enabled using piq.niqe.")
            return ("piq_function", piq)

        if hasattr(piq, "NIQE"):
            metric = piq.NIQE()
            metric = metric.to(device)
            metric.eval()
            print("[info] NIQE enabled using piq.NIQE.")
            return ("piq_module", metric)

        print("[warning] NIQE disabled: installed piq has no NIQE implementation.")
        return None
    except Exception as exc:
        print(f"[warning] NIQE disabled: {exc}")
        return None


def lpips_score(lpips_model, pred, target):
    if lpips_model is None:
        return math.nan

    # LPIPS expects images in [-1, 1].
    pred_lp = pred * 2.0 - 1.0
    target_lp = target * 2.0 - 1.0

    with torch.no_grad():
        value = lpips_model(pred_lp, target_lp)

    return float(value.mean().item())


def niqe_score(niqe_backend, pred):
    if niqe_backend is None:
        return math.nan

    backend_name, metric = niqe_backend

    with torch.no_grad():
        try:
            if backend_name == "pyiqa":
                # pyiqa expects image tensor in [0, 1], NCHW.
                value = metric(pred)
            elif backend_name == "piq_function":
                value = metric.niqe(pred, data_range=1.0, reduction="mean")
            elif backend_name == "piq_module":
                value = metric(pred)
            else:
                return math.nan
        except Exception as exc:
            print(f"[warning] NIQE failed on a batch and was skipped: {exc}")
            return math.nan

    return float(value.mean().item())


def parse_checkpoint_item(item):
    """
    item format:
        name:path

    Example:
        "PG-DASP no prompt:results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth"
    """
    if ":" not in item:
        raise ValueError(
            "Checkpoint item must be in name:path format. "
            f"Got: {item}"
        )

    name, path = item.split(":", 1)
    return name.strip(), path.strip()


def evaluate_checkpoint(
    name,
    checkpoint_path,
    loader,
    device,
    lpips_model,
    niqe_backend,
    strict_load=False,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"[skip] Missing checkpoint: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)

    mode = ckpt.get("mode")
    prompt_mode = ckpt.get("prompt_mode", "full")
    epoch = ckpt.get("epoch", -1)

    if mode is None and "args" in ckpt:
        mode = ckpt["args"].get("mode")
        prompt_mode = ckpt["args"].get("prompt_mode", prompt_mode)

    model = build_model(mode).to(device)

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict_load)
    except Exception as exc:
        print(f"[skip] Could not load {name} from {checkpoint_path}")
        print(f"       Reason: {exc}")
        print("       This usually means the current model_unet.py architecture differs from this checkpoint.")
        return None

    model.eval()

    total = {
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "niqe": 0.0,
    }

    count = 0
    lpips_valid = 0
    niqe_valid = 0

    with torch.no_grad():
        for batch in loader:
            low_rgb = batch["low_rgb"].to(device)
            dasp_input = batch["dasp_input"].to(device)
            target = batch["target"].to(device)

            if mode in ["dasp", "pgdasp", "apgdasp"]:
                dasp_input = apply_prompt_ablation(dasp_input, prompt_mode)

            model_input = get_input(mode, low_rgb, dasp_input)
            pred = model(model_input)

            batch_size = pred.size(0)

            metrics = calculate_metrics(pred, target)
            total["mae"] += metrics["mae"] * batch_size
            total["psnr"] += metrics["psnr"] * batch_size
            total["ssim"] += metrics["ssim"] * batch_size

            lp = lpips_score(lpips_model, pred, target)
            if not math.isnan(lp):
                total["lpips"] += lp * batch_size
                lpips_valid += batch_size

            nq = niqe_score(niqe_backend, pred)
            if not math.isnan(nq):
                total["niqe"] += nq * batch_size
                niqe_valid += batch_size

            count += batch_size

    result = {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "mode": mode,
        "prompt_mode": prompt_mode,
        "mae": total["mae"] / count,
        "psnr": total["psnr"] / count,
        "ssim": total["ssim"] / count,
        "lpips": total["lpips"] / lpips_valid if lpips_valid > 0 else math.nan,
        "niqe": total["niqe"] / niqe_valid if niqe_valid > 0 else math.nan,
    }

    return result


def print_markdown_table(results):
    headers = ["Method", "Epoch", "Mode", "Prompt", "MAE ↓", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "NIQE ↓"]
    print("| " + " | ".join(headers) + " |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|")

    for r in results:
        def fmt(v):
            if isinstance(v, float):
                if math.isnan(v):
                    return "-"
                return f"{v:.4f}"
            return str(v)

        print(
            f"| {r['name']} | {r['epoch']} | {r['mode']} | {r['prompt_mode']} | "
            f"{fmt(r['mae'])} | {fmt(r['psnr'])} | {fmt(r['ssim'])} | "
            f"{fmt(r['lpips'])} | {fmt(r['niqe'])} |"
        )


def save_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "checkpoint",
        "epoch",
        "mode",
        "prompt_mode",
        "mae",
        "psnr",
        "ssim",
        "lpips",
        "niqe",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints with MAE, PSNR, SSIM, LPIPS, and NIQE."
    )

    parser.add_argument("--val-low-dir", type=str, default="data/LOL/lol_dataset/eval15/low")
    parser.add_argument("--val-high-dir", type=str, default="data/LOL/lol_dataset/eval15/high")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default="results/paper_metrics/perceptual_metrics.csv")

    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Checkpoint in name:path format. "
            "Can be used multiple times. If omitted, default paper checkpoints are evaluated."
        ),
    )

    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument("--strict-load", action="store_true", help="Use strict model loading.")

    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)
    print("Device:", device)

    dataset = PairedLowLightDataset(
        low_dir=args.val_low_dir,
        high_dir=args.val_high_dir,
        image_size=(args.height, args.width),
        use_dasp_input=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    lpips_model = try_load_lpips(device)
    niqe_backend = try_load_niqe(device)

    if args.checkpoint:
        checkpoints = [parse_checkpoint_item(item) for item in args.checkpoint]
    else:
        checkpoints = DEFAULT_CHECKPOINTS

    results = []

    for name, path in checkpoints:
        print(f"\nEvaluating: {name}")
        result = evaluate_checkpoint(
            name=name,
            checkpoint_path=path,
            loader=loader,
            device=device,
            lpips_model=lpips_model,
            niqe_backend=niqe_backend,
            strict_load=args.strict_load,
        )
        if result is not None:
            results.append(result)

    print("\nFinal table:")
    print_markdown_table(results)

    save_csv(results, args.output_csv)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
