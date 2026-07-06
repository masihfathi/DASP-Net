# Next Steps for DASP-Net

This document defines the next research and implementation steps after the current IMPROVE paper draft.

## Current conclusion

The current experiments show that the best checkpoint is:

```text
PG-DASP-Net with no prompt
MAE  = 0.0937
PSNR = 20.6343
SSIM = 0.8476
```

The key conclusion is:

```text
Residual-gated feature modulation helps.
Handcrafted prompt maps do not consistently improve enhancement quality.
Simple adaptive prompt weighting is not enough.
```

## Step 1 — Add LPIPS and NIQE

Goal:

```text
Add perceptual and no-reference quality metrics to support the paper discussion.
```

Files:

```text
src/evaluate_paper_metrics.py
requirements_research.txt
```

Run:

```bash
pip install -r requirements_research.txt

python3 src/evaluate_paper_metrics.py \
  --val-low-dir data/LOL/lol_dataset/eval15/low \
  --val-high-dir data/LOL/lol_dataset/eval15/high \
  --height 256 \
  --width 256 \
  --batch-size 1 \
  --output-csv results/paper_metrics/perceptual_metrics.csv
```

Output:

```text
results/paper_metrics/perceptual_metrics.csv
```

## Step 2 — Generate final paper figures

File:

```text
src/make_final_paper_figures.py
```

Run:

```bash
python3 src/make_final_paper_figures.py \
  --metrics-csv results/paper_metrics/perceptual_metrics.csv \
  --output-dir results/paper_figures
```

Output:

```text
results/paper_figures/
```

## Step 3 — Improve adaptive prompt weighting

The next model should not only assign global weights to handcrafted prompt maps.  
A stronger design should use:

```text
spatial prompt confidence
learned prompt selection
prompt reliability estimation
gating strength control at each encoder level
```

Possible name:

```text
LAPG-DASP-Net
Learned Adaptive Prompt-Gated DASP-Net
```

Main idea:

```text
Instead of using fixed handcrafted prompts directly, the network learns a prompt reliability mask and decides where and how much each prompt should affect the features.
```

## Step 4 — Add learned prompt selection

Suggested architecture:

```text
RGB image
   │
   ├── handcrafted prompts: illumination, edge, frequency, noise
   │
   └── learned prompt selector
          ├── spatial weights for each prompt
          ├── no-prompt confidence map
          └── level-wise gate strength
```

Possible formula:

```text
P_selected = Σ wi(x, y) Pi(x, y)
S_l = learned gate strength at encoder level l
F'_l = F_l × (1 + γ_l × S_l × M_l)
```

## Step 5 — Add more datasets

Recommended order:

1. **LOL-v2**  
   Best second paired dataset for low-light enhancement.

2. **SICE**  
   Useful for exposure correction and broader illumination variation.

3. **DICM / LIME / MEF**  
   Useful for qualitative generalization, especially when paired ground truth is unavailable.

Practical plan:

```text
Train on LOL
Evaluate quantitatively on LOL / LOL-v2 if paired data is available
Evaluate qualitatively on DICM, LIME, MEF
```

## Step 6 — Paper-ready final figures

Prepare the following final figures:

1. Method overview diagram
2. Prompt map visualization
3. Main PSNR/SSIM/MAE comparison
4. Prompt ablation figure
5. Adaptive prompt comparison
6. LPIPS and NIQE comparison
7. Final qualitative comparison

## Step 7 — Final paper improvements

Before submission:

```text
Add LPIPS and NIQE table
Add parameter count and runtime table
Add more external qualitative examples
Add comparison with Zero-DCE and Retinex-Net
Prepare anonymous version
Check SCITEPRESS/IMPROVE formatting
```
