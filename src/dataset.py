import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from prompts import read_image_rgb, build_dasp_input


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_image_files(directory: str) -> List[Path]:
    """
    Find all image files inside a directory.
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_files = []

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(directory_path.glob(f"*{ext}"))
        image_files.extend(directory_path.glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {directory}")

    return image_files


def build_pair_list(low_dir: str, high_dir: str) -> List[Tuple[Path, Path]]:
    """
    Build paired list of low-light and normal-light images.

    Matching is done by filename stem.
    Example:
        low/1.png  -> high/1.png
    """
    low_files = find_image_files(low_dir)
    high_files = find_image_files(high_dir)

    high_map: Dict[str, Path] = {
        file.stem: file for file in high_files
    }

    pairs = []

    for low_file in low_files:
        key = low_file.stem

        if key in high_map:
            pairs.append((low_file, high_map[key]))

    if len(pairs) == 0:
        raise RuntimeError(
            "No matching pairs found. Make sure low and high images have the same filenames."
        )

    return pairs


def resize_image(image, image_size: Optional[Tuple[int, int]]):
    """
    Resize image if image_size is provided.

    image_size format:
        (height, width)
    """
    if image_size is None:
        return image

    height, width = image_size

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return resized


class PairedLowLightDataset(Dataset):
    """
    Dataset for paired low-light image enhancement.

    Each sample returns:
        - low_rgb: 3 x H x W
        - dasp_input: 7 x H x W
        - target: 3 x H x W
        - name: image filename
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        image_size: Optional[Tuple[int, int]] = (256, 256),
        use_dasp_input: bool = True,
    ):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.image_size = image_size
        self.use_dasp_input = use_dasp_input

        self.pairs = build_pair_list(low_dir, high_dir)

        print(f"Found {len(self.pairs)} image pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        low_path, high_path = self.pairs[index]

        low_rgb = read_image_rgb(str(low_path))
        high_rgb = read_image_rgb(str(high_path))

        low_rgb = resize_image(low_rgb, self.image_size)
        high_rgb = resize_image(high_rgb, self.image_size)

        low_chw = low_rgb.transpose(2, 0, 1)
        high_chw = high_rgb.transpose(2, 0, 1)

        low_tensor = torch.from_numpy(low_chw).float()
        target_tensor = torch.from_numpy(high_chw).float()

        if self.use_dasp_input:
            dasp_input = build_dasp_input(low_rgb)
            dasp_tensor = torch.from_numpy(dasp_input).float()
        else:
            dasp_tensor = low_tensor

        sample = {
            "low_rgb": low_tensor,
            "dasp_input": dasp_tensor,
            "target": target_tensor,
            "name": low_path.name,
        }

        return sample


def main():
    parser = argparse.ArgumentParser(description="Test paired low-light dataset loader.")

    parser.add_argument("--low-dir", type=str, required=True, help="Path to low-light images.")
    parser.add_argument("--high-dir", type=str, required=True, help="Path to normal-light images.")
    parser.add_argument("--height", type=int, default=256, help="Resize height.")
    parser.add_argument("--width", type=int, default=256, help="Resize width.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")

    args = parser.parse_args()

    dataset = PairedLowLightDataset(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        image_size=(args.height, args.width),
        use_dasp_input=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("Batch loaded successfully.")
    print("Names:", batch["name"])
    print("Low RGB shape:", batch["low_rgb"].shape)
    print("DASP input shape:", batch["dasp_input"].shape)
    print("Target shape:", batch["target"].shape)


if __name__ == "__main__":
    main()