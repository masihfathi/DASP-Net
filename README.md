# DASP-Net Starter Files

## Install

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run prompt map generation

```bash
python src/prompts.py --image data/test_low_light.jpg --out results/prompts
```

Expected output files:

```text
results/prompts/original.png
results/prompts/illumination.png
results/prompts/edge.png
results/prompts/frequency.png
results/prompts/noise.png
```
# DASP-Net
# DASP-Net
