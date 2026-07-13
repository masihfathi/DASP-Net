# Learnable Prompt DASP-Net

Adds three end-to-end variants without modifying the existing `train.py`:

- `refine`: residual prompt correction
- `gate`: spatial reliability gating
- `hybrid`: refinement + reliability gating

## Install

```bash
cd /Users/masihfathi/Desktop/DASP-Net
unzip -o ~/Downloads/DASP_Net_Learnable_Prompt_Tools.zip -d .
chmod +x scripts/run_lpgdasp_all.sh scripts/run_lpgdasp_hybrid_only.sh
```

## Recommended quick experiment

```bash
./scripts/run_lpgdasp_hybrid_only.sh
```

Output:

```text
lpgdasp_hybrid_results.zip
```

## Full ablation

```bash
./scripts/run_lpgdasp_all.sh
```

Output:

```text
lpgdasp_results.zip
```
