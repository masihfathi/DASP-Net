import argparse
import os

import torch
from torchvision.utils import save_image

from model_unet import build_baseline_unet, build_dasp_net
from prompts import read_image_rgb, build_dasp_input


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained checkpoint.")

    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "dasp"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/inference/output.png")

    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    if args.mode == "baseline":
        model = build_baseline_unet()
    else:
        model = build_dasp_net()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    image_rgb = read_image_rgb(args.image)

    if args.mode == "baseline":
        image_chw = image_rgb.transpose(2, 0, 1)
        model_input = torch.from_numpy(image_chw).float().unsqueeze(0)
    else:
        dasp_input = build_dasp_input(image_rgb)
        model_input = torch.from_numpy(dasp_input).float().unsqueeze(0)

    model_input = model_input.to(device)

    with torch.no_grad():
        output = model(model_input)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_image(output.detach().cpu(), args.out)

    print("Output saved to:", args.out)
    print("Checkpoint metrics:", checkpoint.get("val_metrics"))


if __name__ == "__main__":
    main()