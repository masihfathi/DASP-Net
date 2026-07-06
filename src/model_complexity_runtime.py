import argparse
import csv
import sys
import time
from pathlib import Path

import torch

# Make local src imports work when executed as:
# python3 src/model_complexity_runtime.py
sys.path.append(str(Path(__file__).resolve().parent))

from model_unet import (
    build_baseline_unet,
    build_dasp_net,
    build_prompt_gated_dasp_net,
    build_adaptive_prompt_gated_dasp_net,
)


MODEL_BUILDERS = [
    {
        "name": "U-Net",
        "mode": "baseline",
        "prompt_setting": "RGB only",
        "builder": build_baseline_unet,
        "input_channels": 3,
    },
    {
        "name": "Raw DASP-Net",
        "mode": "dasp",
        "prompt_setting": "RGB + all prompts",
        "builder": build_dasp_net,
        "input_channels": 7,
    },
    {
        "name": "PG-DASP-Net",
        "mode": "pgdasp",
        "prompt_setting": "Full prompts",
        "builder": build_prompt_gated_dasp_net,
        "input_channels": 7,
    },
    {
        "name": "PG-DASP-Net",
        "mode": "pgdasp",
        "prompt_setting": "No prompt",
        "builder": build_prompt_gated_dasp_net,
        "input_channels": 7,
    },
    {
        "name": "APG-DASP-Net",
        "mode": "apgdasp",
        "prompt_setting": "Adaptive prompts",
        "builder": build_adaptive_prompt_gated_dasp_net,
        "input_channels": 7,
    },
]


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def try_profile_flops(model, dummy_input):
    """
    FLOPs/MACs are optional because thop may not be installed.

    Install:
        python3 -m pip install thop
    """
    try:
        from thop import profile

        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        # thop returns MACs. Many papers report FLOPs as approximately 2 × MACs.
        flops = 2 * macs
        return macs, flops
    except Exception as exc:
        print(f"[warning] FLOPs/MACs skipped: {exc}")
        return None, None


def benchmark_latency(model, dummy_input, device, warmup=20, repeats=100):
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy_input)

        sync_device(device)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(dummy_input)

        sync_device(device)
        end = time.perf_counter()

    total_time = end - start
    latency_ms = (total_time / repeats) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    return latency_ms, fps


def format_markdown_table(rows):
    headers = [
        "Model",
        "Prompt",
        "Params (M)",
        "Size (MB)",
        "MACs (G)",
        "FLOPs (G)",
        "Latency (ms)",
        "FPS",
    ]

    print("| " + " | ".join(headers) + " |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        def fmt(value, digits=3):
            if value is None:
                return "-"
            if isinstance(value, str):
                return value
            return f"{value:.{digits}f}"

        print(
            f"| {row['name']} | {row['prompt_setting']} | "
            f"{fmt(row['params_m'])} | {fmt(row['size_mb'])} | "
            f"{fmt(row['macs_g'])} | {fmt(row['flops_g'])} | "
            f"{fmt(row['latency_ms'])} | {fmt(row['fps'])} |"
        )


def save_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "mode",
        "prompt_setting",
        "input_shape",
        "params",
        "trainable_params",
        "params_m",
        "size_mb",
        "macs",
        "flops",
        "macs_g",
        "flops_g",
        "latency_ms",
        "fps",
        "device",
        "height",
        "width",
        "batch_size",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Measure parameter count, model size, FLOPs/MACs, latency, and FPS."
    )

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA/MPS is available.")
    parser.add_argument("--skip-flops", action="store_true", help="Skip FLOPs/MACs profiling.")
    parser.add_argument("--output-csv", type=str, default="results/paper_metrics/model_complexity_runtime.csv")

    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)
    print("Device:", device)

    rows = []

    for item in MODEL_BUILDERS:
        print(f"\nProfiling: {item['name']} / {item['prompt_setting']}")

        model = item["builder"]().to(device)
        model.eval()

        dummy_input = torch.randn(
            args.batch_size,
            item["input_channels"],
            args.height,
            args.width,
            device=device,
        )

        params, trainable_params = count_parameters(model)
        size = model_size_mb(model)

        macs, flops = None, None
        if not args.skip_flops:
            # thop is safer on CPU for profiling unsupported ops.
            # If device is MPS, profile a CPU copy to avoid backend issues.
            if device.type == "mps":
                model_for_profile = item["builder"]().cpu().eval()
                dummy_for_profile = torch.randn(
                    args.batch_size,
                    item["input_channels"],
                    args.height,
                    args.width,
                    device="cpu",
                )
                macs, flops = try_profile_flops(model_for_profile, dummy_for_profile)
            else:
                macs, flops = try_profile_flops(model, dummy_input)

        latency_ms, fps = benchmark_latency(
            model=model,
            dummy_input=dummy_input,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        row = {
            "name": item["name"],
            "mode": item["mode"],
            "prompt_setting": item["prompt_setting"],
            "input_shape": f"{args.batch_size}x{item['input_channels']}x{args.height}x{args.width}",
            "params": params,
            "trainable_params": trainable_params,
            "params_m": params / 1e6,
            "size_mb": size,
            "macs": macs,
            "flops": flops,
            "macs_g": macs / 1e9 if macs is not None else None,
            "flops_g": flops / 1e9 if flops is not None else None,
            "latency_ms": latency_ms,
            "fps": fps,
            "device": str(device),
            "height": args.height,
            "width": args.width,
            "batch_size": args.batch_size,
        }

        rows.append(row)

    print("\nFinal table:")
    format_markdown_table(rows)

    save_csv(rows, args.output_csv)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
