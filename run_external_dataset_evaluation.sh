#!/usr/bin/env bash
set -e

cd /Users/masihfathi/Desktop/DASP-Net

mkdir -p results/external_metrics
mkdir -p results/external_outputs
mkdir -p results/external_figures

CKPTS=(
  --checkpoint "U-Net:results/baseline_20epoch/checkpoints/baseline_best.pth"
  --checkpoint "Raw DASP-Net:results/dasp_20epoch/checkpoints/dasp_best.pth"
  --checkpoint "PG-DASP no prompt:results/ablation_none_20epoch/checkpoints/pgdasp_none_best.pth"
  --checkpoint "APG-DASP v3:results/apgdasp_v3_20epoch/checkpoints/apgdasp_best.pth"
)

rename_prefix_in_place () {
  local DIR="$1"
  local PREFIX="$2"

  if [ ! -d "$DIR" ]; then
    echo "[skip] Folder not found: $DIR"
    return
  fi

  echo "Normalizing names in: $DIR"
  echo "Removing prefix: $PREFIX"

  find "$DIR" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
    base="$(basename "$f")"

    if [[ "$base" == "$PREFIX"* ]]; then
      new="${base#$PREFIX}"

      if [ -z "$new" ]; then
        echo "[skip] Empty target name for $base"
        continue
      fi

      target="$DIR/$new"

      if [ "$f" = "$target" ]; then
        continue
      fi

      if [ -e "$target" ]; then
        echo "[warning] Target exists, not overwriting: $target"
        continue
      fi

      mv "$f" "$target"
      echo "[rename] $base -> $new"
    fi
  done
}

normalize_lolv2_names () {
  local LOW="data/external/LOLv2/low"
  local HIGH="data/external/LOLv2/high"

  if [ -d "$LOW" ]; then
    rename_prefix_in_place "$LOW" "low"
  fi

  if [ -d "$HIGH" ]; then
    rename_prefix_in_place "$HIGH" "normal"
  fi

  echo ""
  echo "Checking LOLv2 filename matching..."

  if [ -d "$LOW" ] && [ -d "$HIGH" ]; then
    LOW_COUNT="$(find "$LOW" -maxdepth 1 -type f | wc -l | tr -d ' ')"
    HIGH_COUNT="$(find "$HIGH" -maxdepth 1 -type f | wc -l | tr -d ' ')"

    echo "LOLv2 low count:  $LOW_COUNT"
    echo "LOLv2 high count: $HIGH_COUNT"

    MISMATCH="$(comm -3 <(ls "$LOW" | sort) <(ls "$HIGH" | sort) | head -20 || true)"

    if [ -n "$MISMATCH" ]; then
      echo "[warning] Some LOLv2 filenames still do not match. First mismatches:"
      echo "$MISMATCH"
    else
      echo "[ok] LOLv2 low/high filenames match."
    fi
  fi
}

get_first_three_stems () {
  local DIR="$1"

  find "$DIR" -maxdepth 1 -type f | sort | head -n 3 | while read -r file; do
    basename "$file" | sed 's/\.[^.]*$//'
  done
}

run_paired_dataset () {
  local NAME="$1"
  local LOW="$2"
  local HIGH="$3"
  local OUTCSV="$4"

  if [ -f "$OUTCSV" ]; then
    echo "[skip] Existing metrics found for $NAME: $OUTCSV"
    echo "       Delete this file if you want to re-run evaluation."
    return
  fi

  if [ -d "$LOW" ] && [ -d "$HIGH" ] && [ "$(find "$LOW" -type f | wc -l | tr -d ' ')" -gt 0 ]; then
    echo ""
    echo "=============================="
    echo "Evaluating paired dataset: $NAME"
    echo "=============================="

    python3 src/evaluate_external_datasets.py \
      --dataset-name "$NAME" \
      --low-dir "$LOW" \
      --high-dir "$HIGH" \
      --height 256 \
      --width 256 \
      --batch-size 1 \
      "${CKPTS[@]}" \
      --save-outputs-dir results/external_outputs \
      --output-csv "$OUTCSV"
  else
    echo "[skip] $NAME not found or empty: $LOW / $HIGH"
  fi
}

run_unpaired_dataset () {
  local NAME="$1"
  local LOW="$2"
  local OUTCSV="$3"

  if [ -f "$OUTCSV" ]; then
    echo "[skip] Existing metrics found for $NAME: $OUTCSV"
    echo "       Delete this file if you want to re-run evaluation."
    return
  fi

  if [ -d "$LOW" ] && [ "$(find "$LOW" -type f | wc -l | tr -d ' ')" -gt 0 ]; then
    echo ""
    echo "=============================="
    echo "Evaluating unpaired dataset: $NAME"
    echo "=============================="

    python3 src/evaluate_external_datasets.py \
      --dataset-name "$NAME" \
      --low-dir "$LOW" \
      --height 256 \
      --width 256 \
      --batch-size 1 \
      "${CKPTS[@]}" \
      --save-outputs-dir results/external_outputs \
      --output-csv "$OUTCSV"
  else
    echo "[skip] $NAME not found or empty: $LOW"
  fi
}

make_grid_unpaired () {
  local NAME="$1"
  local LOW="$2"
  local OUTFIG="$3"

  if [ ! -d "$LOW" ]; then
    echo "[skip] no low folder for grid: $LOW"
    return
  fi

  STEMS="$(get_first_three_stems "$LOW" | tr '\n' ' ')"

  if [ -z "$STEMS" ]; then
    echo "[skip] no image stems found for $NAME"
    return
  fi

  echo ""
  echo "Creating qualitative grid for $NAME using stems: $STEMS"

  python3 src/make_external_qualitative_grid.py \
    --title "Qualitative generalization on $NAME" \
    --row-stems $STEMS \
    --column "Low-light:$LOW" \
    --column "U-Net:results/external_outputs/$NAME/U-Net" \
    --column "Raw DASP-Net:results/external_outputs/$NAME/Raw_DASP-Net" \
    --column "PG-DASP no prompt:results/external_outputs/$NAME/PG-DASP_no_prompt" \
    --column "APG-DASP v3:results/external_outputs/$NAME/APG-DASP_v3" \
    --output "$OUTFIG"
}

make_grid_paired () {
  local NAME="$1"
  local LOW="$2"
  local HIGH="$3"
  local OUTFIG="$4"

  if [ ! -d "$LOW" ]; then
    echo "[skip] no low folder for grid: $LOW"
    return
  fi

  STEMS="$(get_first_three_stems "$LOW" | tr '\n' ' ')"

  if [ -z "$STEMS" ]; then
    echo "[skip] no image stems found for $NAME"
    return
  fi

  echo ""
  echo "Creating qualitative grid for $NAME using stems: $STEMS"

  python3 src/make_external_qualitative_grid.py \
    --title "Cross-dataset qualitative comparison on $NAME" \
    --row-stems $STEMS \
    --column "Low-light:$LOW" \
    --column "U-Net:results/external_outputs/$NAME/U-Net" \
    --column "Raw DASP-Net:results/external_outputs/$NAME/Raw_DASP-Net" \
    --column "PG-DASP no prompt:results/external_outputs/$NAME/PG-DASP_no_prompt" \
    --column "APG-DASP v3:results/external_outputs/$NAME/APG-DASP_v3" \
    --column "Reference:$HIGH" \
    --output "$OUTFIG"
}

normalize_lolv2_names

run_paired_dataset "LOLv2" "data/external/LOLv2/low" "data/external/LOLv2/high" "results/external_metrics/lolv2_metrics.csv"
run_paired_dataset "SICE" "data/external/SICE/low" "data/external/SICE/high" "results/external_metrics/sice_metrics.csv"

run_unpaired_dataset "DICM" "data/external/DICM/low" "results/external_metrics/dicm_metrics.csv"
run_unpaired_dataset "LIME" "data/external/LIME/low" "results/external_metrics/lime_metrics.csv"

make_grid_paired "LOLv2" "data/external/LOLv2/low" "data/external/LOLv2/high" "results/external_figures/lolv2_qualitative_grid.png"
make_grid_paired "SICE" "data/external/SICE/low" "data/external/SICE/high" "results/external_figures/sice_qualitative_grid.png"

make_grid_unpaired "DICM" "data/external/DICM/low" "results/external_figures/dicm_qualitative_grid.png"
make_grid_unpaired "LIME" "data/external/LIME/low" "results/external_figures/lime_qualitative_grid.png"

echo ""
echo "=============================="
echo "Done."
echo "Metrics:"
ls -lh results/external_metrics || true
echo ""
echo "Figures:"
ls -lh results/external_figures || true
echo "=============================="
