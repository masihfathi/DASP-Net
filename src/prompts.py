import os
import argparse
from typing import Dict, Tuple

import cv2
import numpy as np


EPS = 1e-8


def normalize_map(x: np.ndarray) -> np.ndarray:
    """
    Normalize a single-channel map to [0, 1].
    """
    x = x.astype(np.float32)
    min_val = np.min(x)
    max_val = np.max(x)

    if max_val - min_val < EPS:
        return np.zeros_like(x, dtype=np.float32)

    return (x - min_val) / (max_val - min_val + EPS)


def read_image_rgb(image_path: str) -> np.ndarray:
    """
    Read image using OpenCV and convert BGR to RGB.
    Output: float32 RGB image in range [0, 1].
    Shape: H x W x 3
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    return image_rgb


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image in [0, 1] to grayscale.
    Output shape: H x W
    """
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def illumination_map(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract illumination map from RGB image.
    """
    illum = np.max(image_rgb, axis=2)
    illum = normalize_map(illum)

    return illum.astype(np.float32)


def edge_map(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract edge map using Sobel operator.
    """
    gray = rgb_to_gray(image_rgb)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge = normalize_map(edge)

    return edge.astype(np.float32)


def frequency_map(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract high-frequency/detail map using Laplacian filter.
    """
    gray = rgb_to_gray(image_rgb)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    freq = np.abs(lap)
    freq = normalize_map(freq)

    return freq.astype(np.float32)


def noise_estimate_map(image_rgb: np.ndarray, blur_kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Estimate noise map using residual between grayscale image and Gaussian-smoothed image.
    """
    gray = rgb_to_gray(image_rgb)

    smooth = cv2.GaussianBlur(gray, blur_kernel, sigmaX=0)
    noise = np.abs(gray - smooth)
    noise = normalize_map(noise)

    return noise.astype(np.float32)


def generate_prompt_maps(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate all DASP prompt maps.
    """
    maps = {
        "illumination": illumination_map(image_rgb),
        "edge": edge_map(image_rgb),
        "frequency": frequency_map(image_rgb),
        "noise": noise_estimate_map(image_rgb),
    }

    return maps


def build_dasp_input(image_rgb: np.ndarray) -> np.ndarray:
    """
    Build 7-channel input for DASP-Net.

    Output:
        dasp_input: 7 x H x W, float32
    """
    maps = generate_prompt_maps(image_rgb)

    rgb_chw = np.transpose(image_rgb, (2, 0, 1))

    illum = maps["illumination"][None, :, :]
    edge = maps["edge"][None, :, :]
    freq = maps["frequency"][None, :, :]
    noise = maps["noise"][None, :, :]

    dasp_input = np.concatenate(
        [rgb_chw, illum, edge, freq, noise],
        axis=0
    )

    return dasp_input.astype(np.float32)


def save_prompt_maps(image_rgb: np.ndarray, output_dir: str) -> None:
    """
    Save original image and generated prompt maps for visual checking.
    """
    os.makedirs(output_dir, exist_ok=True)

    maps = generate_prompt_maps(image_rgb)

    image_uint8 = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(output_dir, "original.png"), image_bgr)

    for name, prompt_map in maps.items():
        map_uint8 = (prompt_map * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), map_uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate DASP-Net prompt maps.")
    parser.add_argument("--image", type=str, required=True, help="Path to input low-light image.")
    parser.add_argument("--out", type=str, default="results/prompts", help="Output directory.")

    args = parser.parse_args()

    image_rgb = read_image_rgb(args.image)

    dasp_input = build_dasp_input(image_rgb)

    print("Image shape:", image_rgb.shape)
    print("DASP input shape:", dasp_input.shape)

    save_prompt_maps(image_rgb, args.out)

    print(f"Prompt maps saved to: {args.out}")


if __name__ == "__main__":
    main()
