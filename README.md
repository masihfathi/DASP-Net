# DASP-Net

**A residual-gated lightweight network for low-light image enhancement and an empirical study of handcrafted visual prompts.**

This repository contains the implementation, training scripts, evaluation tools, prompt-analysis utilities, and reproducibility assets for:

> **A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Empirical Study of Handcrafted Visual Prompts**

The main goal of this project is not simply to introduce another low-light enhancement model, but to study whether handcrafted visual prompts such as illumination, edge, frequency, and noise maps are consistently useful guidance signals for lightweight image-enhancement networks.

---

## Key Idea

DASP-Net investigates a 7-channel input formulation for low-light enhancement:

| Channel | Description |
|---:|---|
| 1–3 | RGB low-light image |
| 4 | Illumination map |
| 5 | Edge map |
| 6 | Frequency/detail map |
| 7 | Noise-estimate map |

The project compares several model families:

| Model | Input | Main idea |
|---|---|---|
| **U-Net Baseline** | RGB only | Standard lightweight encoder-decoder baseline |
| **Raw DASP-Net** | RGB + prompts | Direct concatenation of handcrafted prompts |
| **PG-DASP-Net** | RGB + gated prompts | Residual-gated modulation of prompt features |
| **PG-DASP-Net no prompt** | RGB + zeroed prompts | Tests whether the residual-gated architecture helps without active prompt cues |
| **APG-DASP-Net** | RGB + adaptive prompts | Adaptive prompt-gated variant |

---

## Main Finding

The experiments show that **residual-gated architectures can improve low-light enhancement**, but **handcrafted visual prompts are not universally beneficial**.

On the LOL benchmark, the best lightweight model in this study is **PG-DASP-Net with no active prompt guidance**, suggesting that the residual-gated architecture itself is useful, while handcrafted prompt cues can sometimes introduce bias or amplify unreliable image structures.

A unified comparison with previous methods further shows that a large transformer-based method, **Retinexformer**, achieves stronger absolute reconstruction quality, while DASP-Net remains useful as a lightweight empirical framework for analyzing prompt reliability.

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
│   ├── evaluate_paper_metrics.py
│   ├── evaluate_external_datasets.py
│   ├── evaluate_literature_outputs.py
│   ├── make_qualitative_results.py
│   ├── make_external_qualitative_grid.py
│   ├── make_prompt_map_visualization.py
│   ├── make_prompt_success_failure_cases.py
│   ├── make_final_paper_figures.py
│   ├── model_complexity_runtime.py
│   ├── prepare_sice_dataset.py
│   ├── prepare_lolv2_pairs_by_order.py
│   ├── prepare_paired_train_val_split.py
│   ├── train_external_paired.py
│   └── summarize_external_training.py
│
├── data/
│   ├── LOL/
│   ├── external/
│   └── trainval/
│
├── results/
│   ├── baseline_20epoch/
│   ├── dasp_20epoch/
│   ├── ablation_none_20epoch/
│   ├── apgdasp_v3_20epoch/
│   ├── external_metrics/
│   ├── external_figures/
│   ├── prompt_analysis_figures/
│   ├── prompt_analysis_metrics/
│   └── literature_exact_comparison/
│
├── third_party_benchmark/
│   └── method_outputs/
│
├── scripts/
│   ├── prepare_literature_comparison_folders.sh
│   ├── run_official_zerodce_on_lol.sh
│   ├── run_official_retinexformer_on_lol.sh
│   ├── collect_retinexformer_outputs_and_evaluate.sh
│   └── evaluate_literature_comparison.sh
│
└── README.md
```

---

## Installation

Recommended environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch torchvision numpy pillow scikit-image matplotlib tqdm pandas lpips thop
```

For Apple Silicon, PyTorch MPS is supported when available:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

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

The main LOL experiments use:

| Split | Images |
|---|---:|
| Training | 485 |
| Evaluation | 15 |

---

### LOLv2 and SICE Dataset-Specific Training

Expected deterministic train/validation structure:

```text
data/trainval/LOLv2/
├── train/
│   ├── low/
│   └── high/
└── val/
    ├── low/
    └── high/

data/trainval/SICE/
├── train/
│   ├── low/
│   └── high/
└── val/
    ├── low/
    └── high/
```

SICE is treated as a pseudo-paired exposure benchmark:

- Low image: darkest exposure
- Reference image: exposure closest to target mean luminance
- This is not the same as a standard ground-truth restoration target

---

## Training

### U-Net Baseline

```bash
python3 src/train.py \
  --mode baseline \
  --data-root data/LOL/lol_dataset \
  --output-dir results/baseline_20epoch \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256
```

### Raw DASP-Net

```bash
python3 src/train.py \
  --mode dasp \
  --data-root data/LOL/lol_dataset \
  --output-dir results/dasp_20epoch \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256
```

### PG-DASP-Net

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode full \
  --data-root data/LOL/lol_dataset \
  --output-dir results/pgdasp_v2_20epoch \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256
```

### PG-DASP-Net Without Active Prompts

```bash
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode none \
  --data-root data/LOL/lol_dataset \
  --output-dir results/ablation_none_20epoch \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256
```

### APG-DASP-Net

```bash
python3 src/train.py \
  --mode apgdasp \
  --prompt-mode full \
  --data-root data/LOL/lol_dataset \
  --output-dir results/apgdasp_v3_20epoch \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256
```

---

## Prompt Ablation

The following prompt modes are supported:

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
python3 src/train.py \
  --mode pgdasp \
  --prompt-mode edge \
  --data-root data/LOL/lol_dataset \
  --output-dir results/ablation_edge_20epoch \
  --epochs 20 \
  --batch-size 2 \
  --height 256 \
  --width 256
```

---

## Main LOL Results

Evaluation on LOL `eval15`.

| Method | Epoch | Prompt Setting | MAE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---|---:|---:|---:|---:|
| U-Net Baseline | 15 | RGB only | 0.0960 | 20.3474 | 0.8438 | 0.1444 |
| Raw DASP-Net | 11 | RGB + all prompts | 0.0991 | 20.1702 | 0.8191 | 0.2106 |
| PG-DASP-Net | 9 | Full prompts | 0.1001 | 19.7719 | 0.8315 | 0.1677 |
| **PG-DASP-Net** | **19** | **No active prompt** | **0.0937** | **20.6343** | **0.8476** | **0.1433** |
| APG-DASP-Net v3 | 20 | Adaptive prompts | 0.1033 | 20.0203 | 0.8334 | 0.1452 |

---

## Prompt Ablation Results

| Prompt Setting | Epoch | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Edge only | 16 | 0.0957 | 20.2751 | 0.8401 |
| Frequency only | 12 | 0.1027 | 19.9277 | 0.8338 |
| Illumination only | 17 | 0.0995 | 19.9450 | 0.8427 |
| Noise only | 7 | 0.0979 | 19.8506 | 0.8327 |
| **No active prompt** | **19** | **0.0937** | **20.6343** | **0.8476** |

This result is central to the paper: handcrafted visual prompts do not consistently improve performance, and the best LOL result is obtained when the prompt channels are inactive.

---

## External and Dataset-Specific Evaluation

The project includes external evaluation and dataset-specific training on:

| Dataset | Type | Notes |
|---|---|---|
| LOLv2 | Paired | Used for dataset-specific training and validation |
| SICE | Pseudo-paired | Exposure-based pseudo-reference |
| DICM | Unpaired | Qualitative evaluation |
| LIME | Unpaired | Qualitative evaluation |

Example evaluation:

```bash
python3 src/evaluate_external_datasets.py \
  --dataset-name SICE \
  --low-dir data/external/SICE/low \
  --high-dir data/external/SICE/high \
  --checkpoint "U-Net:results/baseline_20epoch/checkpoints/baseline_best.pth" \
  --checkpoint "PG-DASP no prompt:results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth" \
  --output-csv results/external_metrics/sice_metrics.csv
```

---

## Prompt Map Visualization

Create visualizations for the handcrafted prompt maps:

```bash
python3 src/make_prompt_map_visualization.py \
  --low-image data/trainval/LOLv2/val/low/1.png \
  --reference-image data/trainval/LOLv2/val/high/1.png \
  --title "DASP prompt maps on LOLv2 validation sample" \
  --output results/prompt_analysis_figures/lolv2_prompt_maps.png
```

Generated panels:

```text
Low-light | Illumination | Edge | Frequency | Noise | Reference
```

---

## Prompt Success and Failure Cases

The project can automatically find validation examples where prompt guidance helps or hurts relative to the no-prompt residual-gated model.

```bash
python3 src/make_prompt_success_failure_cases.py \
  --low-dir data/trainval/LOLv2/val/low \
  --high-dir data/trainval/LOLv2/val/high \
  --no-prompt-dir results/training_specific_outputs/LOLv2_trained/PG-DASP_no_prompt_trained_on_LOLv2 \
  --prompt-dir results/training_specific_outputs/LOLv2_trained/APG-DASP_trained_on_LOLv2 \
  --title "Prompt success and failure cases on LOLv2-trained models" \
  --no-prompt-label "PG-DASP no prompt" \
  --prompt-label "APG-DASP" \
  --output results/prompt_analysis_figures/lolv2_prompt_success_failure.png \
  --output-csv results/prompt_analysis_metrics/lolv2_prompt_success_failure.csv
```

---

## Unified Re-Evaluation Against Previous Methods

For a fairer comparison, previous methods can be run on the same LOL `eval15` images and re-evaluated using the same metric pipeline.

The current exact re-evaluation includes:

| Method | MAE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Zero-DCE | 0.1846 | 14.8607 | 0.5624 | 0.3352 | Official output re-evaluated |
| PG-DASP-Net no prompt | 0.0937 | 20.6343 | 0.8476 | 0.1433 | Ours |
| Retinexformer | 0.0468 | 25.1531 | 0.8434 | 0.1314 | Official output re-evaluated |

Interpretation:

- Retinexformer achieves the strongest MAE, PSNR, and LPIPS.
- PG-DASP-Net no prompt achieves SSIM comparable to Retinexformer while remaining lightweight.
- Zero-DCE is useful as a zero-reference baseline, but performs substantially worse on this paired LOL evaluation.
- DASP-Net should be interpreted as a lightweight empirical study of prompt reliability, not as a full SOTA transformer replacement.

---

## Running Previous-Method Comparisons

### Prepare Folders

```bash
./scripts/prepare_literature_comparison_folders.sh
```

### Zero-DCE

```bash
python3 -m pip install opencv-python
./scripts/run_official_zerodce_on_lol.sh
```

### Retinexformer

```bash
./scripts/run_official_retinexformer_on_lol.sh
```

If inference has already run but outputs need to be collected:

```bash
./scripts/collect_retinexformer_outputs_and_evaluate.sh
```

### Evaluate All Available Previous-Method Outputs

```bash
./scripts/evaluate_literature_comparison.sh
```

Output:

```text
literature_exact_comparison_results.zip
results/literature_exact_comparison/lol_summary_metrics.csv
results/literature_exact_comparison/lol_detailed_metrics.csv
results/literature_exact_comparison/figures/literature_comparison_grid.png
```

---

## Model Complexity and Runtime

Measured at 256×256 resolution.

| Model | Prompt | Params (M) | Size (MB) | MACs (G) | FLOPs (G) | Latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| U-Net | RGB only | 7.850 | 29.967 | 14.105 | 28.210 | 11.171 | 89.521 |
| Raw DASP-Net | RGB + all prompts | 7.851 | 29.971 | 14.180 | 28.361 | 11.384 | 87.844 |
| PG-DASP-Net | Full prompts | 8.234 | 31.435 | 14.480 | 28.960 | 13.488 | 74.142 |
| PG-DASP-Net | No prompt | 8.234 | 31.435 | 14.480 | 28.960 | 13.513 | 74.004 |
| APG-DASP-Net | Adaptive prompts | 8.234 | 31.436 | 14.480 | 28.960 | 13.627 | 73.382 |

---

## Inference

Example single-image inference:

```bash
python3 src/inference.py \
  --checkpoint results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth \
  --mode pgdasp \
  --prompt-mode none \
  --input path/to/low_light_image.png \
  --output results/inference_output.png
```

---

## Why Can Handcrafted Prompts Hurt?

Handcrafted prompts are useful only when their assumptions match the target data distribution.

Potential failure modes include:

1. **Illumination maps can be unreliable** when the image contains saturated regions or severe underexposure.
2. **Edge and frequency maps can amplify noise**, especially in dark regions.
3. **Noise-estimate maps can confuse detail with degradation**, causing the network to suppress useful texture.
4. **Prompt distributions shift across datasets**, making fixed handcrafted priors less robust under domain transfer.

The no-prompt PG-DASP result suggests that the residual-gated architecture can be beneficial even without active handcrafted guidance.

---

## Limitations

1. The main experiments use lightweight U-Net-style backbones.
2. SICE is evaluated as a pseudo-paired exposure dataset rather than a standard restoration ground-truth dataset.
3. Handcrafted prompts are fixed and not learned end-to-end.
4. Retinexformer outperforms the proposed lightweight model in absolute reconstruction metrics.
5. The project focuses on prompt reliability and lightweight residual gating, not on claiming state-of-the-art performance.

---

## Paper Assets

The latest manuscript and presentation versions are:

```text
DASP_Net_IMPROVE_Conference_Paper_RetinexformerComparison_Final.pdf
DASP_Net_IMPROVE_Conference_Paper_Named_RetinexformerComparison_Final.pdf
DASP_Net_IMPROVE_Presentation_RetinexformerComparison_Final.pptx
DASP_Net_IMPROVE_Paper_Source_RetinexformerComparison_Final.zip
```

---

## Citation

If you use this code or the experimental framework, please cite:

```bibtex
@inproceedings{fathi2027daspnet,
  title     = {A Residual-Gated Lightweight Network for Low-Light Image Enhancement: An Empirical Study of Handcrafted Visual Prompts},
  author    = {Fathi, Masih},
  booktitle = {International Conference on Image Processing and Vision Engineering},
  year      = {2027}
}
```

---

## Acknowledgements

This project uses and compares against publicly available low-light enhancement benchmarks and previous methods, including Zero-DCE and Retinexformer, for unified re-evaluation on the LOL eval15 split.

---

## License

Add the final license according to your release plan. For academic release, a common choice is:

```text
MIT License
```

or

```text
Apache License 2.0
```

If third-party code is included, keep each third-party repository under its original license.
