import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from prompts import read_image_rgb, build_dasp_input
from model_unet import (
    build_baseline_unet,
    build_dasp_net,
    build_prompt_gated_dasp_net,
)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resize_image(image_rgb, height, width):
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def image_to_tensor_rgb(image_rgb):
    image_chw = image_rgb.transpose(2, 0, 1)
    tensor = torch.from_numpy(image_chw).float().unsqueeze(0)
    return tensor


def tensor_to_pil(tensor):
    """
    Convert tensor B x C x H x W or C x H x W in [0, 1] to PIL image.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu().clamp(0, 1)
    array = tensor.numpy().transpose(1, 2, 0)
    array = (array * 255.0).round().astype(np.uint8)

    return Image.fromarray(array)


def numpy_to_pil(image_rgb):
    image_rgb = np.clip(image_rgb, 0, 1)
    image_uint8 = (image_rgb * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_uint8)


def load_model(mode, checkpoint_path, device):
    if mode == "baseline":
        model = build_baseline_unet()
    elif mode == "dasp":
        model = build_dasp_net()
    elif mode == "pgdasp":
        model = build_prompt_gated_dasp_net()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint.get("val_metrics", None)


def run_model(model, mode, image_rgb, device):
    if mode == "baseline":
        model_input = image_to_tensor_rgb(image_rgb)
    elif mode in ["dasp", "pgdasp"]:
        dasp_input = build_dasp_input(image_rgb)
        model_input = torch.from_numpy(dasp_input).float().unsqueeze(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model_input = model_input.to(device)

    with torch.no_grad():
        output = model(model_input)

    return output


def add_label(image, label, label_height=32):
    """
    Add a simple text label above an image.
    """
    w, h = image.size
    canvas = Image.new("RGB", (w, h + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    x = max((w - text_w) // 2, 0)

    draw.text((x, 8), label, fill=(0, 0, 0), font=font)

    return canvas


def make_comparison_panel(images, labels, gap=8):
    """
    Create horizontal comparison panel.
    """
    labeled_images = [add_label(img, label) for img, label in zip(images, labels)]

    widths = [img.size[0] for img in labeled_images]
    heights = [img.size[1] for img in labeled_images]

    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)

    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    x = 0
    for img in labeled_images:
        canvas.paste(img, (x, 0))
        x += img.size[0] + gap

    return canvas


def find_eval_images(low_dir, names=None, num_samples=5):
    low_dir = Path(low_dir)

    if names:
        files = [low_dir / name for name in names]
    else:
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(low_dir.glob(f"*{ext}"))
            files.extend(low_dir.glob(f"*{ext.upper()}"))
        files = sorted(files)[:num_samples]

    valid_files = []
    for file in files:
        if file.exists():
            valid_files.append(file)
        else:
            print(f"Warning: file not found: {file}")

    if len(valid_files) == 0:
        raise RuntimeError("No valid input images found.")

    return valid_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison figures for DASP-Net paper."
    )

    parser.add_argument("--low-dir", type=str, required=True)
    parser.add_argument("--high-dir", type=str, required=True)

    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--dasp-checkpoint", type=str, required=True)
    parser.add_argument("--pgdasp-checkpoint", type=str, required=True)

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)

    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--names", nargs="*", default=None)

    parser.add_argument("--out-dir", type=str, default="results/qualitative")

    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    baseline_model, baseline_metrics = load_model(
        mode="baseline",
        checkpoint_path=args.baseline_checkpoint,
        device=device,
    )

    dasp_model, dasp_metrics = load_model(
        mode="dasp",
        checkpoint_path=args.dasp_checkpoint,
        device=device,
    )

    pgdasp_model, pgdasp_metrics = load_model(
        mode="pgdasp",
        checkpoint_path=args.pgdasp_checkpoint,
        device=device,
    )

    print("Loaded checkpoints.")
    print("Baseline metrics:", baseline_metrics)
    print("Raw DASP metrics:", dasp_metrics)
    print("PG-DASP v2 metrics:", pgdasp_metrics)

    image_files = find_eval_images(
        low_dir=args.low_dir,
        names=args.names,
        num_samples=args.num_samples,
    )

    labels = [
        "Low-light",
        "U-Net",
        "Raw DASP",
        "PG-DASP v2",
        "Ground Truth",
    ]

    all_panels = []

    for low_path in image_files:
        high_path = Path(args.high_dir) / low_path.name

        if not high_path.exists():
            print(f"Skipping {low_path.name}: target not found.")
            continue

        print("Processing:", low_path.name)

        low_rgb = read_image_rgb(str(low_path))
        high_rgb = read_image_rgb(str(high_path))

        low_rgb = resize_image(low_rgb, args.height, args.width)
        high_rgb = resize_image(high_rgb, args.height, args.width)

        baseline_out = run_model(baseline_model, "baseline", low_rgb, device)
        dasp_out = run_model(dasp_model, "dasp", low_rgb, device)
        pgdasp_out = run_model(pgdasp_model, "pgdasp", low_rgb, device)

        images = [
            numpy_to_pil(low_rgb),
            tensor_to_pil(baseline_out),
            tensor_to_pil(dasp_out),
            tensor_to_pil(pgdasp_out),
            numpy_to_pil(high_rgb),
        ]

        panel = make_comparison_panel(images, labels)
        all_panels.append(panel)

        save_path = Path(args.out_dir) / f"comparison_{low_path.stem}.png"
        panel.save(save_path)
        print("Saved:", save_path)

    if len(all_panels) > 0:
        gap = 12
        max_width = max(panel.size[0] for panel in all_panels)
        total_height = sum(panel.size[1] for panel in all_panels) + gap * (len(all_panels) - 1)

        canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

        y = 0
        for panel in all_panels:
            canvas.paste(panel, (0, y))
            y += panel.size[1] + gap

        combined_path = Path(args.out_dir) / "combined_qualitative_results.png"
        canvas.save(combined_path)
        print("Saved combined figure:", combined_path)


if __name__ == "__main__":
    main()