import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from prompts import build_dasp_input


def read_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def extract_prompt_channels(dasp):
    """
    Expected channel order:
      0 R
      1 G
      2 B
      3 illumination
      4 edge
      5 frequency
      6 noise

    build_dasp_input may return either CHW or HWC.
    """
    arr = np.asarray(dasp, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D DASP input, got shape {arr.shape}")

    # CHW
    if arr.shape[0] == 7:
        return {
            "illumination": arr[3],
            "edge": arr[4],
            "frequency": arr[5],
            "noise": arr[6],
        }

    # HWC
    if arr.shape[-1] == 7:
        return {
            "illumination": arr[:, :, 3],
            "edge": arr[:, :, 4],
            "frequency": arr[:, :, 5],
            "noise": arr[:, :, 6],
        }

    raise ValueError(f"Unsupported DASP shape: {arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Create a visualization of DASP prompt maps.")
    parser.add_argument("--low-image", required=True)
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="DASP handcrafted prompt maps")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    size = (args.width, args.height)

    low = read_rgb(args.low_image, size=size)
    ref = read_rgb(args.reference_image, size=size) if args.reference_image else None

    dasp = build_dasp_input(low)
    maps = extract_prompt_channels(dasp)

    panels = [
        ("Low-light", low, "rgb"),
        ("Illumination", normalize_map(maps["illumination"]), "gray"),
        ("Edge", normalize_map(maps["edge"]), "gray"),
        ("Frequency", normalize_map(maps["frequency"]), "gray"),
        ("Noise", normalize_map(maps["noise"]), "gray"),
    ]

    if ref is not None:
        panels.append(("Reference", ref, "rgb"))

    fig_w = max(12, len(panels) * 2.4)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, 3.2))

    if len(panels) == 1:
        axes = [axes]

    for ax, (name, image, mode) in zip(axes, panels):
        if mode == "rgb":
            ax.imshow(np.clip(image, 0, 1))
        else:
            ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(args.title, fontsize=12)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
