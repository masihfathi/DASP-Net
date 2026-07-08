# Prompt Analysis Figures

This tool creates two types of figures for the paper:

1. Prompt map visualization:
   - Low-light image
   - Illumination map
   - Edge map
   - Frequency map
   - Noise estimate map
   - Reference image

2. Prompt success/failure cases:
   - Finds validation examples where APG-DASP improves over PG-DASP no-prompt.
   - Finds validation examples where APG-DASP performs worse than PG-DASP no-prompt.
   - Uses PSNR difference as the ranking criterion.

## Install

```bash
cd /Users/masihfathi/Desktop/DASP-Net
unzip -o ~/Downloads/DASP_Net_Prompt_Analysis_Tools.zip -d .
chmod +x build_prompt_analysis_figures.sh
```

## Run

```bash
./build_prompt_analysis_figures.sh
```

## Output

```text
prompt_analysis_results.zip
results/prompt_analysis_figures/
results/prompt_analysis_metrics/
```

Upload `prompt_analysis_results.zip` for paper update.
