#!/usr/bin/env bash
set -e

cd /Users/masihfathi/Desktop/DASP-Net

echo "========================================"
echo "Train remaining LPG-DASP variants"
echo "Modes: refine + gate"
echo "Existing hybrid results will be preserved"
echo "========================================"

mkdir -p results/lpgdasp
mkdir -p results/lpgdasp_evaluation
mkdir -p results/lpgdasp_outputs

TRAIN_LOW="data/LOL/lol_dataset/our485/low"
TRAIN_HIGH="data/LOL/lol_dataset/our485/high"
VAL_LOW="data/LOL/lol_dataset/eval15/low"
VAL_HIGH="data/LOL/lol_dataset/eval15/high"

for path in "$TRAIN_LOW" "$TRAIN_HIGH" "$VAL_LOW" "$VAL_HIGH"; do
  if [ ! -d "$path" ]; then
    echo "[error] Missing dataset folder: $path"
    exit 1
  fi
done

for mode in refine gate; do
  echo ""
  echo "=============================="
  echo "Training adapter mode: $mode"
  echo "=============================="

  python3 src/train_lpgdasp.py \
    --adapter-mode "$mode" \
    --train-low-dir "$TRAIN_LOW" \
    --train-high-dir "$TRAIN_HIGH" \
    --val-low-dir "$VAL_LOW" \
    --val-high-dir "$VAL_HIGH" \
    --output-dir "results/lpgdasp/$mode" \
    --epochs 20 \
    --batch-size 2 \
    --height 256 \
    --width 256 \
    --lr 1e-4 \
    --lambda-ssim 0.2 \
    --lambda-grad 0.1 \
    --hidden-channels 32 \
    --initial-residual-scale 0.10 \
    --seed 42

  echo ""
  echo "Evaluating adapter mode: $mode"

  python3 src/evaluate_lpgdasp.py \
    --checkpoint "results/lpgdasp/$mode/checkpoints/lpgdasp_${mode}_best.pth" \
    --adapter-mode "$mode" \
    --low-dir "$VAL_LOW" \
    --high-dir "$VAL_HIGH" \
    --output-dir "results/lpgdasp_outputs/$mode" \
    --output-csv "results/lpgdasp_evaluation/${mode}_metrics.csv" \
    --height 256 \
    --width 256 \
    --batch-size 1
done

if [ ! -f results/lpgdasp_evaluation/hybrid_metrics.csv ]; then
  echo "[error] Existing hybrid metrics not found:"
  echo "results/lpgdasp_evaluation/hybrid_metrics.csv"
  exit 1
fi

python3 src/summarize_lpgdasp.py \
  --input "LPG-DASP refine:results/lpgdasp_evaluation/refine_metrics.csv" \
  --input "LPG-DASP gate:results/lpgdasp_evaluation/gate_metrics.csv" \
  --input "LPG-DASP hybrid:results/lpgdasp_evaluation/hybrid_metrics.csv" \
  --output results/lpgdasp_evaluation/summary.csv

ZIP_NAME="lpgdasp_all_results.zip"
rm -f "$ZIP_NAME"

zip -r "$ZIP_NAME" \
  results/lpgdasp/refine/history.csv \
  results/lpgdasp/refine/config.json \
  results/lpgdasp/gate/history.csv \
  results/lpgdasp/gate/config.json \
  results/lpgdasp/hybrid/history.csv \
  results/lpgdasp/hybrid/config.json \
  results/lpgdasp_evaluation \
  results/lpgdasp_outputs

echo ""
echo "========================================"
echo "Done."
echo "Upload this file:"
echo "$ZIP_NAME"
ls -lh "$ZIP_NAME"
echo "========================================"
