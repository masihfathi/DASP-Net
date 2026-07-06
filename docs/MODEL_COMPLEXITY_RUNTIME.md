# Step: Model Complexity and Runtime

This step adds a table for model parameters, model size, MACs/FLOPs, latency, and FPS.

This is useful for strengthening the conference paper because the proposed model is described as lightweight.

## Files

Copy this file into your project:

```text
src/model_complexity_runtime.py
```

## Optional dependency for MACs/FLOPs

Install `thop`:

```bash
python3 -m pip install thop
```

If `thop` is not installed, the script still computes parameters, model size, latency, and FPS.

## Run

From the project root:

```bash
cd /Users/masihfathi/Desktop/DASP-Net

python3 src/model_complexity_runtime.py \
  --height 256 \
  --width 256 \
  --batch-size 1 \
  --warmup 20 \
  --repeats 100 \
  --output-csv results/paper_metrics/model_complexity_runtime.csv
```

If the Mac MPS backend is unstable, force CPU:

```bash
python3 src/model_complexity_runtime.py \
  --height 256 \
  --width 256 \
  --batch-size 1 \
  --warmup 10 \
  --repeats 50 \
  --cpu \
  --output-csv results/paper_metrics/model_complexity_runtime_cpu.csv
```

## Output

```text
results/paper_metrics/model_complexity_runtime.csv
```

The CSV contains:

```text
params
trainable_params
params_m
size_mb
macs_g
flops_g
latency_ms
fps
```

## Paper interpretation

Important note:

```text
PG-DASP-Net with full prompts and PG-DASP-Net with no prompt use the same architecture.
Therefore, their parameter count and runtime are expected to be nearly identical.
The performance difference comes from prompt ablation, not from a different model size.
```

Recommended paper sentence:

```text
The no-prompt PG-DASP-Net uses the same residual-gated architecture as the full-prompt variant, indicating that the performance gain is not caused by increased model capacity but by avoiding unreliable handcrafted prompt constraints.
```
