# DASP-Net: Residual-Gated Low-Light Image Enhancement

This repository contains the implementation and experimental code for **DASP-Net**, a lightweight low-light image enhancement framework based on residual-gated feature modulation and handcrafted visual prompt analysis.

The project investigates whether handcrafted prompt maps such as illumination, edge, frequency, and noise cues consistently improve low-light enhancement quality. The main empirical finding is that residual-gated architectures can improve enhancement performance, but handcrafted prompts are not always beneficial. In the main LOL benchmark, the best result is obtained by the residual-gated architecture without active prompt guidance.

---

## Paper

**Title:**  
**A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Empirical Study of Handcrafted Visual Prompts**

**Target venue:**  
IMPROVE 2027 — International Conference on Image Processing and Vision Engineering

**Author:**  
Masih Fathi

**Supervisor:**  
Dr. Naghsh

---

## Main Idea

DASP-Net studies low-light image enhancement using two main components:

1. **Handcrafted visual prompts**
   - Illumination map
   - Edge map
   - Frequency map
   - Noise estimate map

2. **Residual-gated prompt modulation**
   - Feature modulation is applied through a residual gate.
   - The gate is initialized to behave close to an identity mapping.
   - This prevents prompt guidance from destabilizing early training.

The project also includes adaptive prompt-gated variants to test whether the model can learn which prompt cues are useful.

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
│   ├── summarize_checkpoints.py
│   ├── evaluate_paper_metrics.py
│   ├── model_complexity_runtime.py
│   ├── evaluate_external_datasets.py
│   ├── make_qualitative_results.py
│   ├── make_final_paper_figures.py
│   ├── make_external_qualitative_grid.py
│   ├── prepare_sice_dataset.py
│   └── prepare_lolv2_pairs_by_order.py
│
├── data/
│   ├── LOL/
│   └── external/
│       ├── LOLv2/
│       ├── SICE/
│       ├── DICM/
│       └── LIME/
│
├── results/
│   ├── baseline_20epoch/
│   ├── dasp_20epoch/
│   ├── pgdasp_v2_20epoch/
│   ├── ablation_none_20epoch/
│   ├── apgdasp_v3_20epoch/
│   ├── external_metrics/
│   ├── external_outputs/
│   └── external_figures/
│
└── README.md
```

---

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install numpy pillow scikit-image matplotlib tqdm pandas lpips
```

Optional packages for extended evaluation:

```bash
python3 -m pip install pyiqa thop
```

On Apple Silicon, the code automatically uses `mps` when available.

---

## Dataset Preparation

### LOL Dataset

Expected structure:

```text
data/LOL/lol_dataset/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

### External Datasets

Expected structure:

```text
data/external/
├── LOLv2/
│   ├── low/
│   └── high/
├── SICE/
│   ├── low/
│   └── high/
├── DICM/
│   └── low/
└── LIME/
    └── low/
```

For LOLv2, filenames must match by stem between low and high folders. For example:

```text
low/00690.png
high/00690.png
```

If the files are named with prefixes such as `low00690.png` and `normal00690.png`, remove the prefixes before evaluation.

---

## Training

### 1. Baseline U-Net

```bash
python3 src/train.py \
  --mode baseline \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/baseline_20epoch
```

### 2. Raw DASP-Net

```bash
python3 src/train.py \
  --mode dasp \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/dasp_20epoch
```

### 3. PG-DASP-Net with full prompts

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode full \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/pgdasp_v2_20epoch
```

### 4. PG-DASP-Net without active prompt guidance

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode none \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/ablation_none_20epoch
```

### 5. APG-DASP-Net

```bash
python3 src/train.py \
  --mode apgdasp \
  --prompt-mode full \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/apgdasp_v3_20epoch
```

---

## Prompt Ablation Study

Run prompt ablations using:

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode illumination \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/ablation_illumination_20epoch
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

Example for edge-only prompt:

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode edge \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256 \
  --output-dir results/ablation_edge_20epoch
```

---

## Summarizing Checkpoints

After training, summarize all best checkpoints:

```bash
python3 src/summarize_checkpoints.py --root results
```

---

## Main LOL Results

Best-checkpoint results on the LOL evaluation set:

| Method | Epoch | Prompt Setting | MAE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---|---:|---:|---:|---:|
| U-Net Baseline | 15 | RGB only | 0.0960 | 20.3474 | 0.8438 | 0.1444 |
| Raw DASP-Net | 11 | RGB + all prompts | 0.0991 | 20.1702 | 0.8191 | 0.2106 |
| PG-DASP-Net | 9 | Full prompts | 0.1001 | 19.7719 | 0.8315 | 0.1677 |
| **PG-DASP-Net** | **19** | **No prompt** | **0.0937** | **20.6343** | **0.8476** | **0.1433** |
| APG-DASP-Net v3 | 20 | Adaptive prompts | 0.1033 | 20.0203 | 0.8334 | 0.1452 |

The best result on the main LOL benchmark is achieved by **PG-DASP-Net without active prompt guidance**.

---

## Prompt Ablation Results

| Prompt Setting | Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| **No prompt** | **19** | **0.0937** | **20.6343** | **0.8476** |
| Illumination only | 17 | 0.0995 | 19.9450 | 0.8427 |
| Edge only | 16 | 0.0957 | 20.2751 | 0.8401 |
| Frequency only | 12 | 0.1027 | 19.9277 | 0.8338 |
| Noise only | 7 | 0.0979 | 19.8506 | 0.8327 |
| Full prompts | 9 | 0.1001 | 19.7719 | 0.8315 |

These results show that handcrafted prompt maps can provide useful cues, but they do not consistently improve reconstruction quality in the main LOL setting.

---

## External Dataset Evaluation

The repository also supports external evaluation on LOLv2, SICE, DICM, and LIME.

Run the full external evaluation script:

```bash
./run_external_dataset_evaluation.sh
```

This creates:

```text
results/external_metrics/
├── lolv2_metrics.csv
├── sice_metrics.csv
├── dicm_metrics.csv
└── lime_metrics.csv

results/external_figures/
├── lolv2_qualitative_grid.png
├── sice_qualitative_grid.png
├── dicm_qualitative_grid.png
└── lime_qualitative_grid.png
```

### External Evaluation Notes

- **LOLv2** is evaluated with paired references.
- **SICE** is used as a pseudo-paired exposure evaluation dataset.
- **DICM** and **LIME** are unpaired datasets, so they are used for qualitative generalization only.
- NIQE is not reported because it was not stable on the current Apple MPS environment.

---

## Model Complexity and Runtime

Run:

```bash
python3 src/model_complexity_runtime.py \
  --height 256 \
  --width 256 \
  --batch-size 1 \
  --warmup 20 \
  --repeats 100 \
  --output-csv results/paper_metrics/model_complexity_runtime.csv
```

Measured on Apple MPS:

| Model | Prompt | Params (M) | Size (MB) | MACs (G) | FLOPs (G) | Latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| U-Net | RGB only | 7.850 | 29.967 | 14.105 | 28.210 | 11.171 | 89.521 |
| Raw DASP-Net | RGB + all prompts | 7.851 | 29.971 | 14.180 | 28.361 | 11.384 | 87.844 |
| PG-DASP-Net | Full prompts | 8.234 | 31.435 | 14.480 | 28.960 | 13.488 | 74.142 |
| PG-DASP-Net | No prompt | 8.234 | 31.435 | 14.480 | 28.960 | 13.513 | 74.004 |
| APG-DASP-Net | Adaptive prompts | 8.234 | 31.436 | 14.480 | 28.960 | 13.627 | 73.382 |

---

## Inference

Example inference command:

```bash
python3 src/inference.py \
  --checkpoint results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth \
  --input path/to/low_light_image.png \
  --output results/inference_output.png
```

---

## Key Findings

1. Residual-gated feature modulation improves low-light enhancement performance.
2. Directly injecting handcrafted prompt maps does not consistently improve performance.
3. The best LOL result is obtained by PG-DASP-Net with no active prompt guidance.
4. Adaptive prompt variants improve some cross-dataset behavior but do not dominate all metrics.
5. External datasets show that prompt usefulness is dataset-dependent.

---

## Recommended Git Ignore

Large datasets, checkpoints, and generated results should not be committed.

Suggested `.gitignore` entries:

```gitignore
# Python
__pycache__/
*.pyc
venv/
.env

# macOS
.DS_Store

# Datasets
data/
Dataset_Part1/
Dataset_Part2/
*.rar
*.zip

# Results and checkpoints
results/
*.pth
*.pt
*.ckpt

# Generated paper files
*.aux
*.log
*.out
*.toc
```

---

## Citation

If you use this repository, please cite the associated paper when it becomes available.

```bibtex
@inproceedings{fathi2027daspnet,
  title     = {A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Empirical Study of Handcrafted Visual Prompts},
  author    = {Fathi, Masih},
  booktitle = {International Conference on Image Processing and Vision Engineering},
  year      = {2027}
}
```

---

## License

This repository is intended for academic and research use.

