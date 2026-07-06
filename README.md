# DASP-Net

**DASP-Net** is a lightweight deep learning framework for **low-light image enhancement**.  
This repository contains the implementation and experimental variants used in the paper:

> **A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Empirical Study of Handcrafted Visual Prompts**

The project studies whether handcrafted visual prompt maps, such as illumination, edge, frequency/detail, and noise cues, consistently improve low-light image enhancement.

---

## Overview

Low-light images often suffer from poor visibility, low contrast, color distortion, noise amplification, and loss of structural detail.  
This project investigates a family of lightweight enhancement networks based on U-Net-style encoder-decoder models and residual prompt-gated feature modulation.

The main finding is that **handcrafted prompt maps are not always beneficial**.  
In the current experiments on the LOL dataset, the best result is achieved by the **residual-gated PG-DASP-Net trained without active prompt guidance**.

---

## Main Contributions

1. A lightweight residual-gated architecture for low-light image enhancement.
2. A handcrafted visual prompt construction pipeline using:
   - illumination maps,
   - edge maps,
   - frequency/detail maps,
   - noise-estimation maps.
3. A systematic ablation study of individual prompt maps.
4. Adaptive prompt-selection variants, including no-prompt selection and controlled residual gating.
5. An empirical analysis showing that handcrafted visual prompts do not consistently improve reconstruction quality on the LOL dataset.

---

## Repository Structure

```text
DASP-Net/
├── src/
│   ├── prompts.py
│   ├── dataset.py
│   ├── model_unet.py
│   ├── metrics.py
│   ├── train.py
│   ├── inference.py
│   ├── load_checkpoint.py
│   ├── make_qualitative_results.py
│   └── summarize_checkpoints.py
├── data/
│   └── LOL/
│       └── lol_dataset/
│           ├── our485/
│           │   ├── low/
│           │   └── high/
│           └── eval15/
│               ├── low/
│               └── high/
├── results/
├── requirements.txt
└── README.md
```

---

## Dataset

The experiments use the **LOL paired low-light dataset**.

Expected directory structure:

```text
data/LOL/lol_dataset/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

- Training set: `our485`
- Evaluation set: `eval15`
- Image size used in experiments: `256 × 256`

---

## Installation

Create and activate a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
torch
torchvision
numpy
opencv-python
pillow
scikit-image
matplotlib
tqdm
```

---

## Prompt Maps

DASP-Net constructs four handcrafted prompt maps from each low-light input image:

| Prompt map | Purpose |
|---|---|
| Illumination map | Captures brightness distribution |
| Edge map | Captures structural boundaries |
| Frequency/detail map | Captures high-frequency texture/detail response |
| Noise map | Estimates local noise-like residuals |

The final DASP input has 7 channels:

```text
[R, G, B, illumination, edge, frequency, noise]
```

---

## Model Variants

### 1. Baseline U-Net

A lightweight RGB-only U-Net.

```bash
--mode baseline
```

### 2. Raw DASP-Net

A U-Net that receives RGB plus all four handcrafted prompt maps.

```bash
--mode dasp
```

### 3. PG-DASP-Net

A prompt-gated DASP-Net using residual feature modulation.

```bash
--mode pgdasp
```

### 4. PG-DASP-Net No-Prompt

The same residual-gated architecture, but with prompt maps removed through ablation.

```bash
--mode pgdasp --prompt-mode none
```

### 5. APG-DASP-Net Variants

Adaptive prompt-gated models that learn prompt weights or no-prompt selection.

```bash
--mode apgdasp
```

---

## Training

### Baseline U-Net

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
  --epochs 20 \
  --output-dir results/baseline_20epoch
```

### Raw DASP-Net

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
  --epochs 20 \
  --output-dir results/dasp_20epoch
```

### PG-DASP-Net with Full Prompts

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode full \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 20 \
  --output-dir results/pgdasp_v2_20epoch
```

### PG-DASP-Net without Prompts

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode none \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 20 \
  --output-dir results/ablation_none_20epoch
```

### APG-DASP-Net

```bash
python3 src/train.py \
  --mode apgdasp \
  --prompt-mode full \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 20 \
  --output-dir results/apgdasp_v3_20epoch
```

---

## Prompt Ablation

To evaluate the contribution of individual prompt maps:

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode edge \
  --train-low-dir data/LOL/lol_dataset/our485/low \
  --train-high-dir data/LOL/lol_dataset/our485/high \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 2 \
  --epochs 20 \
  --output-dir results/ablation_edge_20epoch
```

Available prompt modes:

```text
full
none
illumination
edge
frequency
noise
```

---

## Summarizing Checkpoints

To summarize all best checkpoints:

```bash
python3 src/summarize_checkpoints.py --root results
```

---

## Main Results

Best checkpoints on the LOL evaluation set:

| Method | Prompt setting | Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|---:|---:|
| U-Net Baseline | RGB only | 15 | 0.0960 | 20.3474 | 0.8438 |
| Raw DASP-Net | RGB + all prompts | 11 | 0.0991 | 20.1702 | 0.8191 |
| PG-DASP-Net | Full prompts | 9 | 0.1001 | 19.7719 | 0.8315 |
| **PG-DASP-Net** | **No prompt** | **19** | **0.0937** | **20.6343** | **0.8476** |
| APG-DASP-Net v1 | Adaptive prompts | 16 | 0.0982 | 19.8121 | 0.8397 |
| APG-DASP-Net v2 | No-prompt selection | 20 | 0.1014 | 19.8393 | 0.8382 |
| APG-DASP-Net v3 | Controlled gating | 20 | 0.1033 | 20.0203 | 0.8334 |

---

## Prompt Ablation Results

| Prompt setting | Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| **No prompt** | **19** | **0.0937** | **20.6343** | **0.8476** |
| Illumination only | 17 | 0.0995 | 19.9450 | 0.8427 |
| Edge only | 16 | 0.0957 | 20.2751 | 0.8401 |
| Frequency only | 12 | 0.1027 | 19.9277 | 0.8338 |
| Noise only | 7 | 0.0979 | 19.8506 | 0.8327 |
| Full prompts | 9 | 0.1001 | 19.7719 | 0.8315 |

---

## Key Finding

The best performance is achieved by the residual-gated PG-DASP-Net under the **no-prompt** setting.

This suggests that:

```text
Residual-gated feature modulation is useful.
Handcrafted prompt maps are not universally beneficial.
Simple adaptive prompt selection is insufficient to outperform the no-prompt residual-gated model.
```

---

## Qualitative Results

Representative qualitative comparisons are provided in the paper and presentation assets.

The qualitative figure compares:

```text
Low-light input
U-Net
Raw DASP-Net
PG-DASP-Net No Prompt
APG-DASP-Net v1
APG-DASP-Net v2
APG-DASP-Net v3
Ground Truth
```

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{fathi2027daspnet,
  title={A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Empirical Study of Handcrafted Visual Prompts},
  author={Fathi, Masih},
  booktitle={Proceedings of the International Conference on Image Processing and Vision Engineering},
  year={2027}
}
```

---

## Author

**Masih Fathi**

Repository:

```text
https://github.com/masihfathi/DASP-Net
```

---

## License

This project is released for academic and research use.  
Please check the final repository license before redistribution or commercial use.
