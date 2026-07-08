import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def read_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def psnr(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def find_image_by_stem(folder, stem):
    folder = Path(folder)
    for ext in IMAGE_EXTENSIONS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback case-insensitive / arbitrary extension
    for p in list_images(folder):
        if p.stem == stem:
            return p
    return None


def metric_table(low_dir, high_dir, no_prompt_dir, prompt_dir):
    low_files = list_images(low_dir)
    rows = []

    for low_path in low_files:
        stem = low_path.stem
        ref_path = find_image_by_stem(high_dir, stem)
        no_path = find_image_by_stem(no_prompt_dir, stem)
        pr_path = find_image_by_stem(prompt_dir, stem)

        if ref_path is None or no_path is None or pr_path is None:
            continue

        # Use prediction size as common size.
        pred_size = Image.open(no_path).convert("RGB").size

        ref = read_rgb(ref_path, size=pred_size)
        no = read_rgb(no_path, size=pred_size)
        pr = read_rgb(pr_path, size=pred_size)

        no_psnr = psnr(no, ref)
        pr_psnr = psnr(pr, ref)
        no_mae = mae(no, ref)
        pr_mae = mae(pr, ref)

        rows.append({
            "stem": stem,
            "low_path": str(low_path),
            "reference_path": str(ref_path),
            "no_prompt_path": str(no_path),
            "prompt_path": str(pr_path),
            "no_prompt_psnr": no_psnr,
            "prompt_psnr": pr_psnr,
            "delta_psnr_prompt_minus_no": pr_psnr - no_psnr,
            "no_prompt_mae": no_mae,
            "prompt_mae": pr_mae,
            "delta_mae_prompt_minus_no": pr_mae - no_mae,
        })

    return rows


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stem",
        "low_path",
        "reference_path",
        "no_prompt_path",
        "prompt_path",
        "no_prompt_psnr",
        "prompt_psnr",
        "delta_psnr_prompt_minus_no",
        "no_prompt_mae",
        "prompt_mae",
        "delta_mae_prompt_minus_no",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_cases(help_cases, hurt_cases, output, title, no_prompt_label, prompt_label):
    cases = []
    for r in help_cases:
        cases.append(("Prompt helps", r))
    for r in hurt_cases:
        cases.append(("Prompt hurts", r))

    if not cases:
        raise RuntimeError("No cases available for plotting.")

    nrows = len(cases)
    ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Low-light", no_prompt_label, prompt_label, "Reference"]

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10)

    for r_idx, (case_name, row) in enumerate(cases):
        no_path = Path(row["no_prompt_path"])
        pred_size = Image.open(no_path).convert("RGB").size

        images = [
            read_rgb(row["low_path"], size=pred_size),
            read_rgb(row["no_prompt_path"], size=pred_size),
            read_rgb(row["prompt_path"], size=pred_size),
            read_rgb(row["reference_path"], size=pred_size),
        ]

        delta = row["delta_psnr_prompt_minus_no"]
        label = (
            f"{case_name}\n"
            f"{row['stem']}\n"
            f"ΔPSNR={delta:+.2f} dB"
        )

        for c_idx, img in enumerate(images):
            ax = axes[r_idx, c_idx]
            ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")
            if c_idx == 0:
                ax.set_ylabel(label, fontsize=9)

        axes[r_idx, 1].set_xlabel(f"PSNR={row['no_prompt_psnr']:.2f}", fontsize=8)
        axes[r_idx, 2].set_xlabel(f"PSNR={row['prompt_psnr']:.2f}", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Find success/failure cases where prompt-guided output helps or hurts compared with no-prompt output."
    )
    parser.add_argument("--low-dir", required=True)
    parser.add_argument("--high-dir", required=True)
    parser.add_argument("--no-prompt-dir", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--title", default="Prompt success and failure cases")
    parser.add_argument("--no-prompt-label", default="No-prompt")
    parser.add_argument("--prompt-label", default="Prompt-guided")
    parser.add_argument("--num-help", type=int, default=2)
    parser.add_argument("--num-hurt", type=int, default=2)
    args = parser.parse_args()

    rows = metric_table(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        no_prompt_dir=args.no_prompt_dir,
        prompt_dir=args.prompt_dir,
    )

    if not rows:
        raise FileNotFoundError("No matched low/reference/no-prompt/prompt images were found.")

    rows = sorted(rows, key=lambda r: r["delta_psnr_prompt_minus_no"], reverse=True)

    help_cases = rows[:args.num_help]
    hurt_cases = list(reversed(rows[-args.num_hurt:]))

    save_csv(rows, args.output_csv)
    plot_cases(
        help_cases=help_cases,
        hurt_cases=hurt_cases,
        output=args.output,
        title=args.title,
        no_prompt_label=args.no_prompt_label,
        prompt_label=args.prompt_label,
    )

    print("\nTop prompt-help cases:")
    for r in help_cases:
        print(f"{r['stem']}: ΔPSNR={r['delta_psnr_prompt_minus_no']:+.4f}")

    print("\nTop prompt-hurt cases:")
    for r in hurt_cases:
        print(f"{r['stem']}: ΔPSNR={r['delta_psnr_prompt_minus_no']:+.4f}")


if __name__ == "__main__":
    main()
