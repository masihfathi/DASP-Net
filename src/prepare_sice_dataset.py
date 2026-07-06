import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sequence_sort_key(path: Path):
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def mean_luminance(path: Path) -> float:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    y = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(y.mean())


def choose_pair(image_paths, target_luminance=0.55):
    """
    Choose:
    - low: darkest exposure in the sequence
    - high/reference: exposure closest to target_luminance

    This avoids always choosing the brightest image, which may be over-exposed.
    """
    scored = [(p, mean_luminance(p)) for p in image_paths]
    scored = sorted(scored, key=lambda x: x[1])

    low_path, low_lum = scored[0]

    if len(scored) == 1:
        raise ValueError("Only one exposure found.")

    high_path, high_lum = min(scored[1:], key=lambda x: abs(x[1] - target_luminance))

    return low_path, high_path, low_lum, high_lum, scored


def copy_as_png(src: Path, dst: Path, resize=None):
    img = Image.open(src).convert("RGB")

    if resize is not None:
        w, h = resize
        img = img.resize((w, h), Image.BICUBIC)

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a pseudo-paired SICE evaluation split from multi-exposure sequences."
    )

    parser.add_argument(
        "--sice-root",
        type=str,
        required=True,
        help="Path to Dataset_Part1 or Dataset_Part2.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/external/SICE",
        help="Output root containing low/ and high/.",
    )

    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="",
        help="Prefix added to output filenames, e.g. part1_ or part2_. This avoids overwriting.",
    )

    parser.add_argument(
        "--target-luminance",
        type=float,
        default=0.55,
        help="Target mean luminance for selecting the reference exposure. Default: 0.55",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Optional limit for quick testing. 0 means use all sequences.",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optional resize to square size. 0 means keep original resolution.",
    )

    args = parser.parse_args()

    sice_root = Path(args.sice_root)
    output_root = Path(args.output_root)
    low_out = output_root / "low"
    high_out = output_root / "high"

    safe_prefix = args.filename_prefix.replace("/", "_").replace(" ", "_")
    report_name = f"sice_pair_selection_report_{safe_prefix.rstrip('_') or sice_root.name}.csv"
    report_path = output_root / report_name

    low_out.mkdir(parents=True, exist_ok=True)
    high_out.mkdir(parents=True, exist_ok=True)

    if not sice_root.exists():
        raise FileNotFoundError(f"SICE root not found: {sice_root}")

    sequence_dirs = sorted(
        [p for p in sice_root.iterdir() if p.is_dir()],
        key=sequence_sort_key,
    )

    if args.max_sequences and args.max_sequences > 0:
        sequence_dirs = sequence_dirs[: args.max_sequences]

    resize = None
    if args.resize and args.resize > 0:
        resize = (args.resize, args.resize)

    rows = []
    processed = 0
    skipped = 0

    for seq_dir in sequence_dirs:
        images = list_images(seq_dir)

        if len(images) < 2:
            skipped += 1
            print(f"[skip] {seq_dir.name}: fewer than 2 images")
            continue

        try:
            low_path, high_path, low_lum, high_lum, scored = choose_pair(
                images,
                target_luminance=args.target_luminance,
            )
        except Exception as exc:
            skipped += 1
            print(f"[skip] {seq_dir.name}: {exc}")
            continue

        out_name = f"{safe_prefix}{seq_dir.name}.png"

        copy_as_png(low_path, low_out / out_name, resize=resize)
        copy_as_png(high_path, high_out / out_name, resize=resize)

        rows.append(
            {
                "sequence": seq_dir.name,
                "output_name": out_name,
                "num_images": len(images),
                "low_file": str(low_path),
                "high_file": str(high_path),
                "low_luminance": low_lum,
                "high_luminance": high_lum,
                "all_luminances": "; ".join([f"{p.name}:{lum:.4f}" for p, lum in scored]),
            }
        )

        processed += 1

        print(
            f"[ok] {seq_dir.name}: "
            f"low={low_path.name} ({low_lum:.3f}) | "
            f"high={high_path.name} ({high_lum:.3f}) -> {out_name}"
        )

    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sequence",
            "output_name",
            "num_images",
            "low_file",
            "high_file",
            "low_luminance",
            "high_luminance",
            "all_luminances",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Low dir:   {low_out}")
    print(f"High dir:  {high_out}")
    print(f"Report:    {report_path}")


if __name__ == "__main__":
    main()
