from pathlib import Path
from datasets import load_dataset

output_root = Path("data/LOL-HF")
high_dir = output_root / "high"
low_dir = output_root / "low"

high_dir.mkdir(parents=True, exist_ok=True)
low_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("geekyrakshit/LoL-Dataset", split="train")

high_count = 0
low_count = 0

for i, item in enumerate(dataset):
    image = item["image"]
    label = item["label"]

    # در این دیتاست:
    # 0 = high
    # 1 = low
    if label == 0:
        save_path = high_dir / f"{high_count:04d}.png"
        high_count += 1
    else:
        save_path = low_dir / f"{low_count:04d}.png"
        low_count += 1

    image.save(save_path)

print("Download and export finished.")
print("High images:", high_count)
print("Low images:", low_count)
print("Saved to:", output_root)