import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def copy_or_link(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation splits for paired low-light datasets."
    )
    parser.add_argument("--low-dir", required=True, help="Input low-light image folder.")
    parser.add_argument("--high-dir", required=True, help="Input reference/high-light image folder.")
    parser.add_argument("--output-root", required=True, help="Output root, e.g. data/trainval/LOLv2.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    low_dir = Path(args.low_dir)
    high_dir = Path(args.high_dir)
    output_root = Path(args.output_root)

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    low_files = list_images(low_dir)
    high_files = list_images(high_dir)
    high_by_stem = {p.stem: p for p in high_files}

    pairs = []
    for low_path in low_files:
        high_path = high_by_stem.get(low_path.stem)
        if high_path is not None:
            pairs.append((low_path.stem, low_path, high_path))

    if not pairs:
        raise FileNotFoundError(f"No paired images matched by filename stem between {low_dir} and {high_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    val_count = max(1, int(round(len(pairs) * args.val_ratio)))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    if not train_pairs:
        raise ValueError("Train split is empty. Use a smaller --val-ratio.")

    manifest_rows = []

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for stem, low_path, high_path in split_pairs:
            low_dst = output_root / split_name / "low" / low_path.name
            high_dst = output_root / split_name / "high" / high_path.name

            copy_or_link(low_path, low_dst, args.mode)
            copy_or_link(high_path, high_dst, args.mode)

            manifest_rows.append({
                "split": split_name,
                "stem": stem,
                "low_source": str(low_path),
                "high_source": str(high_path),
                "low_output": str(low_dst),
                "high_output": str(high_dst),
            })

    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "stem", "low_source", "high_source", "low_output", "high_output"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Dataset split saved to: {output_root}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    import os
    main()
