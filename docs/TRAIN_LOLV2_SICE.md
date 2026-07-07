# Training on LOLv2 and SICE

This package adds training support for external paired datasets instead of evaluation-only usage.

## Files

Copy these files into the project:

```bash
cp src/prepare_paired_train_val_split.py /Users/masihfathi/Desktop/DASP-Net/src/
cp src/train_external_paired.py /Users/masihfathi/Desktop/DASP-Net/src/
cp src/summarize_external_training.py /Users/masihfathi/Desktop/DASP-Net/src/
cp run_train_lolv2_sice.sh /Users/masihfathi/Desktop/DASP-Net/
chmod +x /Users/masihfathi/Desktop/DASP-Net/run_train_lolv2_sice.sh
```

## Run

```bash
cd /Users/masihfathi/Desktop/DASP-Net
./run_train_lolv2_sice.sh
```

## Summarize

```bash
python3 src/summarize_external_training.py \
  --root results \
  --output-csv results/external_training_summary.csv
```

## Experimental design

The script creates deterministic train/validation splits:

- LOLv2: 80% train / 20% validation
- SICE: 85% train / 15% validation

The validation subsets are held out from training. This is stronger than evaluation-only testing because it shows how the models behave when trained on each dataset.

## Important paper wording

For SICE, report the result as a pseudo-paired exposure evaluation, not as a standard ground-truth restoration benchmark.

Suggested wording:

> To further investigate dataset-specific behavior, we train the compared models directly on LOLv2 and on a pseudo-paired SICE exposure split. The SICE experiment is treated as an exposure-normalization study because the reference images are selected from exposure sequences rather than from a separately captured ground-truth restoration target.
