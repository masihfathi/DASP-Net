# DASP-Net

**DASP-Net** is a lightweight low-light image enhancement research project.

Current research direction:

> A residual-gated lightweight network for low-light image enhancement, with an experimental analysis of hand-crafted visual prompt maps.

The project investigates whether hand-crafted prompt maps can help a lightweight network improve low-light images.

Prompt maps studied in this repository:

- Illumination map
- Edge map
- Frequency/detail map
- Noise estimation map

The main experimental finding so far is:

> Hand-crafted prompt maps are not always beneficial when directly or uniformly injected into the network. In the current experiments, the best result is obtained by **PG-DASP-Net v2 without prompt maps**, which suggests that the residual-gated architecture itself improves reconstruction, while prompt maps require adaptive selection or weighting.

---

## Project Goal

Low-light images often suffer from poor visibility, low contrast, noise amplification, color distortion, and loss of fine details.

This repository compares:

1. Baseline U-Net
2. Raw DASP-Net
3. PG-DASP-Net
4. PG-DASP-Net v2
5. Prompt ablation variants

---

## Method Overview

### Baseline U-Net

```text
RGB image -> U-Net -> Enhanced RGB image
```

### Raw DASP-Net

Raw DASP-Net concatenates RGB and four prompt maps:

```text
RGB + Illumination + Edge + Frequency + Noise = 7-channel input
```

Then the 7-channel input is passed directly to a U-Net.

### PG-DASP-Net v2

PG-DASP-Net v2 uses a residual identity gate:

```text
F' = F × (1 + gamma × M)
```

where:

- `F` is the RGB feature
- `M` is the modulation map from the prompt branch
- `gamma` is a learnable parameter initialized to zero

At the start of training:

```text
gamma = 0
F' ≈ F
```

This lets the model start close to the baseline and gradually learn how much the guidance branch should affect RGB features.

---

## Repository Structure

```text
DASP-Net/
  data/
    LOL/
      lol_dataset/
        our485/
          low/
          high/
        eval15/
          low/
          high/

  src/
    prompts.py
    dataset.py
    model_unet.py
    metrics.py
    train.py
    inference.py
    load_checkpoint.py
    summarize_checkpoints.py
    make_qualitative_results.py
    download_lol_zip.py
    download_lol_hf.py

  results/
  requirements.txt
  README.md
  .gitignore
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
opencv-python
numpy
torch
torchvision
huggingface_hub
pillow
```

---

## Download LOL Dataset

```bash
python3 src/download_lol_zip.py
```

Expected structure:

```text
data/LOL/lol_dataset/our485/low
data/LOL/lol_dataset/our485/high
data/LOL/lol_dataset/eval15/low
data/LOL/lol_dataset/eval15/high
```

Dataset split used:

```text
Training set: 485 paired images
Testing set:  15 paired images
```

---

## Generate Prompt Maps

```bash
python3 src/prompts.py   --image data/test_low_light.jpg   --out results/prompts
```

Expected output:

```text
results/prompts/original.png
results/prompts/illumination.png
results/prompts/edge.png
results/prompts/frequency.png
results/prompts/noise.png
```

---

## Test Dataset Loader

### Train set

```bash
python3 src/dataset.py   --low-dir data/LOL/lol_dataset/our485/low   --high-dir data/LOL/lol_dataset/our485/high   --height 256   --width 256   --batch-size 2
```

Expected output:

```text
Found 485 image pairs.
Batch loaded successfully.
Low RGB shape: torch.Size([2, 3, 256, 256])
DASP input shape: torch.Size([2, 7, 256, 256])
Target shape: torch.Size([2, 3, 256, 256])
```

### Test set

```bash
python3 src/dataset.py   --low-dir data/LOL/lol_dataset/eval15/low   --high-dir data/LOL/lol_dataset/eval15/high   --height 256   --width 256   --batch-size 1
```

Expected output:

```text
Found 15 image pairs.
Batch loaded successfully.
Low RGB shape: torch.Size([1, 3, 256, 256])
DASP input shape: torch.Size([1, 7, 256, 256])
Target shape: torch.Size([1, 3, 256, 256])
```

---

## Test Model Architecture

```bash
python3 src/model_unet.py
```

The script tests:

- Baseline U-Net
- Raw DASP-Net
- PG-DASP-Net v2

Expected output shape for all models:

```text
torch.Size([1, 3, 256, 256])
```

---

## Metrics

Implemented in:

```text
src/metrics.py
```

Metrics:

- MAE
- PSNR
- SSIM

Test:

```bash
python3 src/metrics.py
```

---

## Training

### Baseline U-Net

```bash
python3 src/train.py   --mode baseline   --train-low-dir data/LOL/lol_dataset/our485/low   --train-high-dir data/LOL/lol_dataset/our485/high   --val-low-dir data/LOL/lol_dataset/eval15/low   --val-high-dir data/LOL/lol_dataset/eval15/high   --height 256   --width 256   --batch-size 2   --epochs 20   --output-dir results/baseline_20epoch
```

### Raw DASP-Net

```bash
python3 src/train.py   --mode dasp   --train-low-dir data/LOL/lol_dataset/our485/low   --train-high-dir data/LOL/lol_dataset/our485/high   --val-low-dir data/LOL/lol_dataset/eval15/low   --val-high-dir data/LOL/lol_dataset/eval15/high   --height 256   --width 256   --batch-size 2   --epochs 20   --output-dir results/dasp_20epoch
```

### PG-DASP-Net v2 with all prompt maps

```bash
python3 src/train.py   --mode pgdasp   --prompt-mode full   --train-low-dir data/LOL/lol_dataset/our485/low   --train-high-dir data/LOL/lol_dataset/our485/high   --val-low-dir data/LOL/lol_dataset/eval15/low   --val-high-dir data/LOL/lol_dataset/eval15/high   --height 256   --width 256   --batch-size 2   --epochs 20   --output-dir results/pgdasp_v2_20epoch
```

### PG-DASP-Net v2 without prompt maps

```bash
python3 src/train.py   --mode pgdasp   --prompt-mode none   --train-low-dir data/LOL/lol_dataset/our485/low   --train-high-dir data/LOL/lol_dataset/our485/high   --val-low-dir data/LOL/lol_dataset/eval15/low   --val-high-dir data/LOL/lol_dataset/eval15/high   --height 256   --width 256   --batch-size 2   --epochs 20   --output-dir results/ablation_none_20epoch
```

---

## Prompt Ablation

The `--prompt-mode` argument controls which prompt maps are active.

Available options:

```text
full
none
illumination
edge
frequency
noise
```

Example:

```bash
python3 src/train.py   --mode pgdasp   --prompt-mode edge   --train-low-dir data/LOL/lol_dataset/our485/low   --train-high-dir data/LOL/lol_dataset/our485/high   --val-low-dir data/LOL/lol_dataset/eval15/low   --val-high-dir data/LOL/lol_dataset/eval15/high   --height 256   --width 256   --batch-size 2   --epochs 20   --output-dir results/ablation_edge_20epoch
```

---

## Checkpoint Files

`.pth` files are PyTorch checkpoint files and should not be opened directly as text.

Inspect a checkpoint:

```bash
python3 src/load_checkpoint.py   --checkpoint results/baseline_20epoch/checkpoints/baseline_best.pth
```

Summarize all best checkpoints:

```bash
python3 src/summarize_checkpoints.py --root results
```

---

## Current Best Results

The following results are based on the best checkpoint of each experiment, selected by the highest validation PSNR.

| Model / Setting | Best Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Baseline U-Net | 15 | 0.0960 | 20.3474 | 0.8438 |
| Raw DASP-Net | 11 | 0.0991 | 20.1702 | 0.8191 |
| PG-DASP-Net v2, full prompts | 9 | 0.1001 | 19.7719 | 0.8315 |
| PG-DASP-Net v2, no prompts | 19 | **0.0937** | **20.6343** | **0.8476** |

---

## Prompt Ablation Results

| Prompt Setting | Best Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| No prompt maps | 19 | **0.0937** | **20.6343** | **0.8476** |
| Illumination only | 17 | 0.0995 | 19.9450 | 0.8427 |
| Edge only | 16 | 0.0957 | 20.2751 | 0.8401 |
| Frequency only | 12 | 0.1027 | 19.9277 | 0.8338 |
| Noise only | 7 | 0.0979 | 19.8506 | 0.8327 |
| Full prompts | 9 | 0.1001 | 19.7719 | 0.8315 |

---

## Qualitative Results

Generate comparison figures:

```bash
python3 src/make_qualitative_results.py   --low-dir data/LOL/lol_dataset/eval15/low   --high-dir data/LOL/lol_dataset/eval15/high   --baseline-checkpoint results/baseline_20epoch/checkpoints/baseline_best.pth   --dasp-checkpoint results/dasp_20epoch/checkpoints/dasp_best.pth   --pgdasp-checkpoint results/pgdasp_v2_20epoch/checkpoints/pgdasp_best.pth   --height 256   --width 256   --num-samples 5   --out-dir results/qualitative
```

Output:

```text
results/qualitative/combined_qualitative_results.png
```

Each row:

```text
Low-light input | U-Net | Raw DASP | PG-DASP v2 | Ground Truth
```

---

## Inference

### Baseline U-Net

```bash
python3 src/inference.py   --mode baseline   --checkpoint results/baseline_20epoch/checkpoints/baseline_best.pth   --image data/LOL/lol_dataset/eval15/low/146.png   --out results/inference/baseline_146.png
```

### Raw DASP-Net

```bash
python3 src/inference.py   --mode dasp   --checkpoint results/dasp_20epoch/checkpoints/dasp_best.pth   --image data/LOL/lol_dataset/eval15/low/146.png   --out results/inference/dasp_146.png
```

### PG-DASP-Net

```bash
python3 src/inference.py   --mode pgdasp   --checkpoint results/pgdasp_v2_20epoch/checkpoints/pgdasp_best.pth   --image data/LOL/lol_dataset/eval15/low/146.png   --out results/inference/pgdasp_146.png
```

---

## Git Ignore

Recommended `.gitignore`:

```gitignore
venv/
.venv/
env/
ENV/

__pycache__/
*.pyc

data/
results/
*.pth

.DS_Store
```

Remove already-tracked large files:

```bash
git rm -r --cached venv data results
git add .gitignore
git commit -m "Ignore data, results, and checkpoints"
```

---

## Research Direction

Current paper title:

```text
A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Analysis of Hand-Crafted Visual Prompts
```

Persian title:

```text
یک شبکه سبک با gate باقی‌مانده برای بهبود تصاویر کم‌نور: تحلیل اثر نقشه‌های راهنمای دستی
```

Main conclusion:

> Residual-gated feature modulation improves low-light image enhancement, while hand-crafted prompt maps do not consistently improve performance in the current setup.

---

## Citation

```bibtex
@misc{fathi2026daspnet,
  author = {Fathi, Masih},
  title = {A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Analysis of Hand-Crafted Visual Prompts},
  year = {2026},
  howpublished = {\url{https://github.com/masihfathi/DASP-Net}}
}
```

---

## Status

This project is under active development.

Implemented:

- Prompt map generation
- LOL dataset downloader
- Paired LOL dataset loader
- Baseline U-Net
- Raw DASP-Net
- PG-DASP-Net v2
- Residual identity gate
- MAE, PSNR, and SSIM
- Training script
- Prompt ablation
- Checkpoint summarization
- Qualitative result generation
- Inference script

Next steps:

- Improve adaptive prompt weighting
- Add learned prompt selection
- Add more datasets
- Add LPIPS and NIQE
- Prepare paper-ready final figures
