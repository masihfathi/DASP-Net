from pathlib import Path
import zipfile
from huggingface_hub import hf_hub_download


def main():
    output_dir = Path("data/LOL")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LOL dataset zip from Hugging Face...")

    zip_path = hf_hub_download(
        repo_id="geekyrakshit/LoL-Dataset",
        filename="lol_dataset.zip",
        repo_type="dataset",
        local_dir="data/downloads",
    )

    print("Downloaded to:", zip_path)
    print("Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Done.")
    print("Extracted to:", output_dir)

    print("\nFolder structure:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_dir():
            print(path)


if __name__ == "__main__":
    main()