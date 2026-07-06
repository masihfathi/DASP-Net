import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def read_rgb(path):
    return Image.open(path).convert("RGB")


def crop_or_resize(img, size=(256, 256)):
    return img.resize(size, Image.BICUBIC)


def parse_column(item):
    """
    Format:
        label:path

    Example:
        "Low-light:data/DICM/input"
        "PG-DASP:results/external_outputs/DICM/PG-DASP_no_prompt"
    """
    if ":" not in item:
        raise ValueError("Column must be in label:path format.")

    label, path = item.split(":", 1)
    return label.strip(), Path(path.strip())


def find_by_stem(folder, stem):
    folder = Path(folder)
    for ext in IMAGE_EXTENSIONS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p

    # Fall back: search recursively
    matches = [p for p in folder.rglob("*") if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Create qualitative comparison grid for external datasets.")
    parser.add_argument("--row-stems", nargs="+", required=True, help="Image filename stems to show as rows.")
    parser.add_argument("--column", action="append", required=True, help="Column in label:path format.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default="External qualitative comparison")
    parser.add_argument("--panel-size", type=int, default=256)

    args = parser.parse_args()

    columns = [parse_column(item) for item in args.column]
    row_stems = args.row_stems

    n_rows = len(row_stems)
    n_cols = len(columns)

    fig_w = max(8, n_cols * 2.2)
    fig_h = max(4, n_rows * 2.2 + 0.8)

    plt.figure(figsize=(fig_w, fig_h))

    for r, stem in enumerate(row_stems):
        for c, (label, folder) in enumerate(columns):
            img_path = find_by_stem(folder, stem)
            ax = plt.subplot(n_rows, n_cols, r * n_cols + c + 1)

            if img_path is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.axis("off")
                continue

            img = crop_or_resize(read_rgb(img_path), (args.panel_size, args.panel_size))
            ax.imshow(img)
            ax.axis("off")

            if r == 0:
                ax.set_title(label, fontsize=10)

            if c == 0:
                ax.text(
                    -0.04,
                    0.5,
                    stem,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )

    plt.suptitle(args.title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
