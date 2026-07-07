#!/usr/bin/env bash
set -e

cd /Users/masihfathi/Desktop/DASP-Net

python3 src/prepare_paired_train_val_split.py \
  --low-dir data/external/LOLv2/low \
  --high-dir data/external/LOLv2/high \
  --output-root data/trainval/LOLv2 \
  --val-ratio 0.20 \
  --seed 42 \
  --overwrite

python3 src/prepare_paired_train_val_split.py \
  --low-dir data/external/SICE/low \
  --high-dir data/external/SICE/high \
  --output-root data/trainval/SICE \
  --val-ratio 0.15 \
  --seed 42 \
  --overwrite

train_if_needed () {
  local DATASET="$1"
  local MODE="$2"
  local PROMPT="$3"
  local OUTDIR="$4"
  local TRAIN_LOW="$5"
  local TRAIN_HIGH="$6"
  local VAL_LOW="$7"
  local VAL_HIGH="$8"

  local CKPT="$OUTDIR/checkpoints/${MODE}_${PROMPT}_best.pth"

  if [ -f "$CKPT" ]; then
    echo ""
    echo "[skip] Existing checkpoint found:"
    echo "$CKPT"
    echo "       Delete this checkpoint if you want to retrain."
    return
  fi

  echo ""
  echo "=============================="
  echo "Training $DATASET | $MODE | $PROMPT"
  echo "=============================="

  python3 src/train_external_paired.py \
    --dataset-name "$DATASET" \
    --train-low-dir "$TRAIN_LOW" \
    --train-high-dir "$TRAIN_HIGH" \
    --val-low-dir "$VAL_LOW" \
    --val-high-dir "$VAL_HIGH" \
    --mode "$MODE" \
    --prompt-mode "$PROMPT" \
    --epochs 20 \
    --batch-size 2 \
    --height 256 \
    --width 256 \
    --output-dir "$OUTDIR"
}

# LOLv2
train_if_needed "LOLv2" "baseline" "full" \
  "results/train_lolv2/baseline" \
  "data/trainval/LOLv2/train/low" \
  "data/trainval/LOLv2/train/high" \
  "data/trainval/LOLv2/val/low" \
  "data/trainval/LOLv2/val/high"

train_if_needed "LOLv2" "dasp" "full" \
  "results/train_lolv2/dasp_full" \
  "data/trainval/LOLv2/train/low" \
  "data/trainval/LOLv2/train/high" \
  "data/trainval/LOLv2/val/low" \
  "data/trainval/LOLv2/val/high"

train_if_needed "LOLv2" "pgdasp" "none" \
  "results/train_lolv2/pgdasp_none" \
  "data/trainval/LOLv2/train/low" \
  "data/trainval/LOLv2/train/high" \
  "data/trainval/LOLv2/val/low" \
  "data/trainval/LOLv2/val/high"

train_if_needed "LOLv2" "apgdasp" "full" \
  "results/train_lolv2/apgdasp_full" \
  "data/trainval/LOLv2/train/low" \
  "data/trainval/LOLv2/train/high" \
  "data/trainval/LOLv2/val/low" \
  "data/trainval/LOLv2/val/high"

# SICE
train_if_needed "SICE" "baseline" "full" \
  "results/train_sice/baseline" \
  "data/trainval/SICE/train/low" \
  "data/trainval/SICE/train/high" \
  "data/trainval/SICE/val/low" \
  "data/trainval/SICE/val/high"

train_if_needed "SICE" "dasp" "full" \
  "results/train_sice/dasp_full" \
  "data/trainval/SICE/train/low" \
  "data/trainval/SICE/train/high" \
  "data/trainval/SICE/val/low" \
  "data/trainval/SICE/val/high"

train_if_needed "SICE" "pgdasp" "none" \
  "results/train_sice/pgdasp_none" \
  "data/trainval/SICE/train/low" \
  "data/trainval/SICE/train/high" \
  "data/trainval/SICE/val/low" \
  "data/trainval/SICE/val/high"

train_if_needed "SICE" "apgdasp" "full" \
  "results/train_sice/apgdasp_full" \
  "data/trainval/SICE/train/low" \
  "data/trainval/SICE/train/high" \
  "data/trainval/SICE/val/low" \
  "data/trainval/SICE/val/high"

echo ""
echo "Training finished."
echo "LOLv2 histories:"
find results/train_lolv2 -name history.csv -print

echo ""
echo "SICE histories:"
find results/train_sice -name history.csv -print

echo ""
echo "Best checkpoints:"
find results/train_lolv2 results/train_sice -name "*_best.pth" -print

echo ""
echo "Summary:"
python3 src/summarize_external_training.py \
  --root results \
  --output-csv results/external_training_summary.csv
