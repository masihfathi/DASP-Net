import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    print("Checkpoint loaded successfully.")
    print("Keys:")
    for key in checkpoint.keys():
        print("-", key)

    print("\nMode:", checkpoint.get("mode"))
    print("Epoch:", checkpoint.get("epoch"))
    print("Validation metrics:", checkpoint.get("val_metrics"))


if __name__ == "__main__":
    main()