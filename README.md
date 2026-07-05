# DASP-Net

**DASP-Net** stands for **Degradation-Aware Spatial-Frequency Prompting Network**.

This repository contains a lightweight deep learning framework for **low-light image enhancement**. The main idea is to improve a simple U-Net model by adding several guidance maps extracted directly from the input image.

Instead of using only the RGB image, DASP-Net uses:

- RGB image
- Illumination map
- Edge map
- Frequency/detail map
- Noise estimation map

Therefore, the proposed model receives a **7-channel input** and predicts a **3-channel enhanced RGB image**.

---

## Project Goal

Low-light images often suffer from:

- poor visibility
- low contrast
- noise amplification
- color distortion
- loss of fine details

Many deep learning methods improve image quality, but they often require complex architectures or high computational cost. This project aims to build a simple and lightweight method that improves low-light images by using additional spatial and frequency-aware prompt maps while keeping the model architecture close to a standard U-Net.

---

## Method Overview

Given a low-light RGB image, DASP-Net first extracts four guidance maps.

### 1. Illumination Map

The illumination map captures the brightness distribution of the image.

### 2. Edge Map

The edge map is extracted using the Sobel operator and helps preserve object boundaries.

### 3. Frequency/Detail Map

The frequency/detail map is extracted using the Laplacian operator and highlights high-frequency details.

### 4. Noise Estimation Map

The noise estimation map is computed using the residual between the grayscale image and its Gaussian-smoothed version.

The final input is:

```text
RGB + Illumination + Edge + Frequency + Noise = 7 channels
```

The model output is:

```text
7-channel input -> Lightweight U-Net -> 3-channel enhanced image
```

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
    download_lol_zip.py
    download_lol_hf.py

  results/
    prompts/
    baseline_test/
      checkpoints/
      samples/
    dasp_test/
      checkpoints/
      samples/
    inference/

  requirements.txt
  README.md
  .gitignore
```

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file should contain:

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

The LOL dataset can be downloaded from Hugging Face using the provided script.

Run:

```bash
python3 src/download_lol_zip.py
```

After extraction, the expected structure is:

```text
data/LOL/lol_dataset/our485/low
data/LOL/lol_dataset/our485/high
data/LOL/lol_dataset/eval15/low
data/LOL/lol_dataset/eval15/high
```

The dataset split is:

```text
Training set: 485 paired images
Testing set:  15 paired images
```

---

## Generate Prompt Maps

To generate DASP guidance maps for a sample low-light image:

```bash
python3 src/prompts.py \
  --image data/test_low_light.jpg \
  --out results/prompts
```

Expected output files:

```text
results/prompts/original.png
results/prompts/illumination.png
results/prompts/edge.png
results/prompts/frequency.png
results/prompts/noise.png
```

These files are useful for visually checking whether the prompt maps are generated correctly.

---

## Test Dataset Loader

### Train Set

```bash
python3 src/dataset.py \
  --low-dir data/LOL/lol_dataset/our485/low \
  --high-dir data/LOL/lol_dataset/our485/high \
  --height 256 \
  --width 256 \
  --batch-size 2
```

Expected output:

```text
Found 485 image pairs.
Batch loaded successfully.
Low RGB shape: torch.Size([2, 3, 256, 256])
DASP input shape: torch.Size([2, 7, 256, 256])
Target shape: torch.Size([2, 3, 256, 256])
```

### Test Set

```bash
python3 src/dataset.py \
  --low-dir data/LOL/lol_dataset/eval15/low \
  --high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 1
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

Run:

```bash
python3 src/model_unet.py
```

Expected output:

```text
Device: cpu

Baseline U-Net
Input shape: torch.Size([1, 3, 256, 256])
Output shape: torch.Size([1, 3, 256, 256])
Parameters: 7849667

DASP-Net
Input shape: torch.Size([1, 7, 256, 256])
Output shape: torch.Size([1, 3, 256, 256])
Parameters: 7850819
```

Current parameter count:

```text
Baseline U-Net: 7,849,667 parameters
DASP-Net:       7,850,819 parameters
```

DASP-Net adds only:

```text
1,152 extra parameters
```

This means the proposed model adds four guidance maps with almost no increase in model complexity.

---

## Evaluation Metrics

The project currently uses three evaluation metrics:

- **MAE**: Mean Absolute Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index Measure

These metrics are implemented in:

```text
src/metrics.py
```

To test the metrics file:

```bash
python3 src/metrics.py
```

Expected output:

```text
Metrics test:
MAE: ...
PSNR: ...
SSIM: ...
```

---

## Train Baseline U-Net

Run a quick one-epoch baseline test:

```bash
python3 src/train.py \
  --mode baseline \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 1 \
  --output-dir results/baseline_test
```

After training, the following files should be generated:

```text
results/baseline_test/checkpoints/baseline_best.pth
results/baseline_test/checkpoints/baseline_last.pth
results/baseline_test/samples/baseline_epoch_001.png
```

The sample image contains:

```text
low-light input | enhanced output | ground truth
```

---

## Train DASP-Net

Run a quick one-epoch DASP-Net test:

```bash
python3 src/train.py \
  --mode dasp \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 1 \
  --output-dir results/dasp_test
```

After training, the following files should be generated:

```text
results/dasp_test/checkpoints/dasp_best.pth
results/dasp_test/checkpoints/dasp_last.pth
results/dasp_test/samples/dasp_epoch_001.png
```

---

## Checkpoint Files

The `.pth` files are PyTorch checkpoint files.

Examples:

```text
baseline_best.pth
baseline_last.pth
dasp_best.pth
dasp_last.pth
```

These files are binary model files and should not be opened directly in a text editor such as VS Code. They should be loaded using `torch.load`.

---

## Inspect a Checkpoint

To inspect a saved checkpoint:

```bash
python3 src/load_checkpoint.py \
  --checkpoint results/baseline_test/checkpoints/baseline_best.pth
```

Expected output:

```text
Checkpoint loaded successfully.
Keys:
- epoch
- mode
- model_state_dict
- optimizer_state_dict
- val_metrics

Mode: baseline
Epoch: 1
Validation metrics: {'mae': ..., 'psnr': ..., 'ssim': ...}
```

For DASP-Net:

```bash
python3 src/load_checkpoint.py \
  --checkpoint results/dasp_test/checkpoints/dasp_best.pth
```

---

## Run Inference

After training, inference can be performed on a low-light image using a saved checkpoint.

### Baseline U-Net Inference

```bash
python3 src/inference.py \
  --mode baseline \
  --checkpoint results/baseline_test/checkpoints/baseline_best.pth \
  --image data/LOL/lol_dataset/eval15/low/146.png \
  --out results/inference/baseline_146.png
```

Output:

```text
results/inference/baseline_146.png
```

### DASP-Net Inference

```bash
python3 src/inference.py \
  --mode dasp \
  --checkpoint results/dasp_test/checkpoints/dasp_best.pth \
  --image data/LOL/lol_dataset/eval15/low/146.png \
  --out results/inference/dasp_146.png
```

Output:

```text
results/inference/dasp_146.png
```

---

## Output Files

Training and inference generate the following files:

```text
results/
  baseline_test/
    checkpoints/
      baseline_best.pth
      baseline_last.pth
    samples/
      baseline_epoch_001.png

  dasp_test/
    checkpoints/
      dasp_best.pth
      dasp_last.pth
    samples/
      dasp_epoch_001.png

  inference/
    baseline_146.png
    dasp_146.png
```

Image files such as `.png` can be opened normally.  
Checkpoint files such as `.pth` should be loaded with PyTorch.

---

## Recommended Quick Test Order

A good order for testing the project is:

```bash
python3 src/prompts.py \
  --image data/test_low_light.jpg \
  --out results/prompts
```

```bash
python3 src/model_unet.py
```

```bash
python3 src/dataset.py \
  --low-dir data/LOL/lol_dataset/our485/low \
  --high-dir data/LOL/lol_dataset/our485/high \
  --height 256 \
  --width 256 \
  --batch-size 2
```

```bash
python3 src/metrics.py
```

```bash
python3 src/train.py \
  --mode baseline \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 1 \
  --output-dir results/baseline_test
```

```bash
python3 src/train.py \
  --mode dasp \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 1 \
  --output-dir results/dasp_test
```

---

## Git Ignore

The following folders and files should not be committed to GitHub:

```text
venv/
data/
results/
*.pth
__pycache__/
```

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

If these files were already added to Git, remove them from tracking:

```bash
git rm -r --cached venv data results
git add .gitignore
git commit -m "Ignore data, results, and model checkpoints"
```

---

## Current Implementation Status

| Component | Status |
|---|---|
| Prompt map generation | Implemented |
| LOL dataset downloader | Implemented |
| LOL paired dataset loader | Implemented |
| Baseline U-Net | Implemented |
| DASP-Net 7-channel model | Implemented |
| MAE metric | Implemented |
| PSNR metric | Implemented |
| SSIM metric | Implemented |
| Training script | Implemented |
| Checkpoint loading | Implemented |
| Inference script | Implemented |
| Full test script | Planned |
| Ablation study | Planned |
| Result table generation | Planned |
| Paper-ready figures | Planned |

---

## Paper Direction

This repository is part of a research project on low-light image enhancement.

The current paper title is:

```text
DASP-Net: Degradation-Aware Spatial-Frequency Prompting Network for Low-Light Image Enhancement
```

Main contribution:

```text
A lightweight low-light image enhancement model that uses illumination, edge, frequency, and noise-aware prompt maps to guide a simple U-Net with minimal additional parameters.
```

---

## Citation

If you use this repository, please cite it as:

```bibtex
@misc{fathi2026daspnet,
  author = {Fathi, Masih},
  title = {DASP-Net: Degradation-Aware Spatial-Frequency Prompting Network for Low-Light Image Enhancement},
  year = {2026},
  howpublished = {\url{https://github.com/masihfathi/DASP-Net}}
}
```

---

## License

This project is currently under active development.  
A license file will be added later.

---

## Status

This project is under active development.

Current focus:

```text
1. Train baseline U-Net
2. Train DASP-Net
3. Compare PSNR, SSIM, and MAE
4. Generate qualitative comparison images
5. Prepare experiment tables for the paper
```