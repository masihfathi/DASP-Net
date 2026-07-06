import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


MAIN_RESULTS = [
    {"Method": "U-Net", "MAE": 0.0960, "PSNR": 20.3474, "SSIM": 0.8438},
    {"Method": "Raw DASP", "MAE": 0.0991, "PSNR": 20.1702, "SSIM": 0.8191},
    {"Method": "PG-DASP full", "MAE": 0.1001, "PSNR": 19.7719, "SSIM": 0.8315},
    {"Method": "PG-DASP none", "MAE": 0.0937, "PSNR": 20.6343, "SSIM": 0.8476},
    {"Method": "APG v1", "MAE": 0.0982, "PSNR": 19.8121, "SSIM": 0.8397},
    {"Method": "APG v2", "MAE": 0.1014, "PSNR": 19.8393, "SSIM": 0.8382},
    {"Method": "APG v3", "MAE": 0.1033, "PSNR": 20.0203, "SSIM": 0.8334},
]

ABLATION_RESULTS = [
    {"Prompt": "No prompt", "MAE": 0.0937, "PSNR": 20.6343, "SSIM": 0.8476},
    {"Prompt": "Illumination", "MAE": 0.0995, "PSNR": 19.9450, "SSIM": 0.8427},
    {"Prompt": "Edge", "MAE": 0.0957, "PSNR": 20.2751, "SSIM": 0.8401},
    {"Prompt": "Frequency", "MAE": 0.1027, "PSNR": 19.9277, "SSIM": 0.8338},
    {"Prompt": "Noise", "MAE": 0.0979, "PSNR": 19.8506, "SSIM": 0.8327},
    {"Prompt": "Full", "MAE": 0.1001, "PSNR": 19.7719, "SSIM": 0.8315},
]

ADAPTIVE_RESULTS = [
    {"Method": "APG v1", "MAE": 0.0982, "PSNR": 19.8121, "SSIM": 0.8397},
    {"Method": "APG v2", "MAE": 0.1014, "PSNR": 19.8393, "SSIM": 0.8382},
    {"Method": "APG v3", "MAE": 0.1033, "PSNR": 20.0203, "SSIM": 0.8334},
    {"Method": "PG-DASP none", "MAE": 0.0937, "PSNR": 20.6343, "SSIM": 0.8476},
]


def save_bar(df, x_col, y_col, title, ylabel, output_path, rotate=25):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    bars = ax.bar(df[x_col], df[y_col])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}" if value < 1 else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_builtin_figures(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.DataFrame(MAIN_RESULTS)
    ablation_df = pd.DataFrame(ABLATION_RESULTS)
    adaptive_df = pd.DataFrame(ADAPTIVE_RESULTS)

    save_bar(
        main_df,
        "Method",
        "PSNR",
        "Main comparison on LOL",
        "PSNR (dB)",
        output_dir / "main_results_psnr_paper.png",
    )

    save_bar(
        main_df,
        "Method",
        "SSIM",
        "Main comparison on LOL",
        "SSIM",
        output_dir / "main_results_ssim_paper.png",
    )

    save_bar(
        main_df,
        "Method",
        "MAE",
        "Main comparison on LOL",
        "MAE",
        output_dir / "main_results_mae_paper.png",
    )

    save_bar(
        ablation_df,
        "Prompt",
        "PSNR",
        "Prompt ablation study",
        "PSNR (dB)",
        output_dir / "prompt_ablation_psnr_paper.png",
    )

    save_bar(
        adaptive_df,
        "Method",
        "PSNR",
        "Adaptive prompt variants",
        "PSNR (dB)",
        output_dir / "adaptive_variants_psnr_paper.png",
    )


def make_perceptual_figures(csv_path, output_dir):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[skip] Perceptual CSV not found: {csv_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "lpips" in df.columns and df["lpips"].notna().any():
        save_bar(
            df,
            "name",
            "lpips",
            "LPIPS comparison",
            "LPIPS ↓",
            output_dir / "lpips_comparison_paper.png",
        )

    if "niqe" in df.columns and df["niqe"].notna().any():
        save_bar(
            df,
            "name",
            "niqe",
            "NIQE comparison",
            "NIQE ↓",
            output_dir / "niqe_comparison_paper.png",
        )


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--output-dir", type=str, default="results/paper_figures")
    parser.add_argument("--metrics-csv", type=str, default="results/paper_metrics/perceptual_metrics.csv")

    args = parser.parse_args()

    make_builtin_figures(args.output_dir)
    make_perceptual_figures(args.metrics_csv, args.output_dir)

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
