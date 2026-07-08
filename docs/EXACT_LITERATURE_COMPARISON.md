# Exact comparison with previous papers

## Install

```bash
cd /Users/masihfathi/Desktop/DASP-Net
unzip -o ~/Downloads/DASP_Net_Literature_Exact_Comparison_Tools.zip -d .
chmod +x scripts/*.sh
```

## Prepare benchmark folders

```bash
./scripts/prepare_literature_comparison_folders.sh
```

## Run Zero-DCE official code

```bash
python3 -m pip install opencv-python
./scripts/run_official_zerodce_on_lol.sh
```

## Retinexformer

Use the official repo and pretrained LOL-v1 weight. Official command:

```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/RetinexFormer_LOL_v1.yml \
  --weights pretrained_weights/LOL_v1.pth \
  --dataset LOL_v1
```

Copy output images into:

```text
third_party_benchmark/method_outputs/LOL/Retinexformer
```

## SNR-Aware

Use the official repo and pretrained LOL-v1 checkpoint. Official command:

```bash
python test_LOLv1_v2_real.py -opt options/test/LOLv1.yml
```

Copy output images into:

```text
third_party_benchmark/method_outputs/LOL/SNR-Aware
```

## RetinexNet

Older TensorFlow code. If it runs, copy enhanced LOL eval15 outputs into:

```text
third_party_benchmark/method_outputs/LOL/RetinexNet
```

## Evaluate

```bash
./scripts/evaluate_literature_comparison.sh
```

Upload:

```text
literature_exact_comparison_results.zip
```
