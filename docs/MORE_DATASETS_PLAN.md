# Adding More Datasets to DASP-Net

This document gives the recommended order for adding more datasets to the paper.

## Important distinction

There are two dataset types:

### 1. Paired datasets

These have low-light input and reference/ground-truth images.

For paired datasets we can report:

```text
MAE
PSNR
SSIM
LPIPS
```

Recommended paired datasets:

```text
LOL-v2
SICE / SICE Part 1
```

### 2. Unpaired / no-reference datasets

These have only low-light images and no ground truth.

For these datasets we cannot honestly report PSNR, SSIM, MAE, or LPIPS.

For unpaired datasets we can report:

```text
NIQE, if available
qualitative comparison
visual generalization
```

Recommended unpaired datasets:

```text
DICM
LIME
NPE
MEF
VV
```

## Recommended paper strategy

For this paper, use:

```text
LOL          quantitative + qualitative
LOL-v2       quantitative + qualitative
SICE         quantitative if paired reference is prepared, otherwise qualitative
DICM/LIME    qualitative generalization + NIQE if working
```

## Folder structure

Create this structure:

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

The script pairs images by filename stem. Example:

```text
low/001.png
high/001.png
```

## Evaluate paired dataset

Example for LOL-v2:

```bash
python3 src/evaluate_external_datasets.py \
  --dataset-name LOL-v2 \
  --low-dir data/external/LOLv2/low \
  --high-dir data/external/LOLv2/high \
  --height 256 \
  --width 256 \
  --batch-size 1 \
  --checkpoint "U-Net:results/baseline_20epoch/checkpoints/baseline_best.pth" \
  --checkpoint "Raw DASP-Net:results/dasp_20epoch/checkpoints/dasp_best.pth" \
  --checkpoint "PG-DASP no prompt:results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth" \
  --checkpoint "APG-DASP v3:results/apgdasp_v3_20epoch/checkpoints/apgdasp_best.pth" \
  --save-outputs-dir results/external_outputs \
  --output-csv results/external_metrics/lolv2_metrics.csv
```

## Evaluate unpaired dataset

Example for DICM:

```bash
python3 src/evaluate_external_datasets.py \
  --dataset-name DICM \
  --low-dir data/external/DICM/low \
  --height 256 \
  --width 256 \
  --batch-size 1 \
  --checkpoint "U-Net:results/baseline_20epoch/checkpoints/baseline_best.pth" \
  --checkpoint "Raw DASP-Net:results/dasp_20epoch/checkpoints/dasp_best.pth" \
  --checkpoint "PG-DASP no prompt:results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth" \
  --checkpoint "APG-DASP v3:results/apgdasp_v3_20epoch/checkpoints/apgdasp_best.pth" \
  --save-outputs-dir results/external_outputs \
  --output-csv results/external_metrics/dicm_metrics.csv
```

## Make qualitative grid

After running inference with `--save-outputs-dir`, choose a few image stems and create a qualitative figure.

Example:

```bash
python3 src/make_external_qualitative_grid.py \
  --title "Qualitative generalization on DICM" \
  --row-stems 01 02 03 \
  --column "Low-light:data/external/DICM/low" \
  --column "U-Net:results/external_outputs/DICM/U-Net" \
  --column "Raw DASP-Net:results/external_outputs/DICM/Raw_DASP-Net" \
  --column "PG-DASP no prompt:results/external_outputs/DICM/PG-DASP_no_prompt" \
  --column "APG-DASP v3:results/external_outputs/DICM/APG-DASP_v3" \
  --output results/external_figures/dicm_qualitative_grid.png
```

## Recommended text for the paper

```text
To examine generalization beyond the original LOL evaluation split, we additionally evaluate the trained models on external low-light datasets. For paired datasets, we report MAE, PSNR, SSIM, and LPIPS. For unpaired datasets, where reference images are unavailable, we restrict the analysis to qualitative comparisons and no-reference quality assessment when available. This avoids reporting reference-based metrics in settings where ground truth is not defined.
```
