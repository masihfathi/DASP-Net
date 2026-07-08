# Retinexformer official run for DASP-Net comparison

This package runs the official Retinexformer implementation on the same LOL eval15 split used in DASP-Net.

## Install

```bash
cd /Users/masihfathi/Desktop/DASP-Net

unzip -o ~/Downloads/DASP_Net_Retinexformer_Run_Tools.zip -d .

chmod +x scripts/run_official_retinexformer_on_lol.sh
chmod +x scripts/collect_retinexformer_outputs_and_evaluate.sh
```

## Run

```bash
./scripts/run_official_retinexformer_on_lol.sh
```

Expected final output:

```text
third_party_benchmark/method_outputs/LOL/Retinexformer
literature_exact_comparison_results.zip
```

## If pretrained weight download fails

Download the official pretrained weights folder manually:

```text
https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV
```

Put this file:

```text
third_party_repos/Retinexformer/pretrained_weights/LOL_v1.pth
```

Then re-run:

```bash
./scripts/run_official_retinexformer_on_lol.sh
```

## If inference runs but output copying fails

Run:

```bash
./scripts/collect_retinexformer_outputs_and_evaluate.sh
```

Then upload:

```text
literature_exact_comparison_results.zip
```

## Notes

We intentionally do not pass `--GT_mean`, because that setting uses the ground-truth mean and is not the main fair setting for our unified comparison.
