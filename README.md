# HRM LLM Add-on

This repository contains a lightweight trainer and Reasoning Gym wrapper for experimenting with the hybrid HRM + LLM architecture. It provides tooling to build answer-only datasets from Reasoning Gym tasks, train a controller with optional HRM injection, and run proxy evaluations.

## Getting started

1. Install dependencies (including [reasoning-gym](https://github.com/openai/reasoning-gym)) into your Python environment.
2. Optionally export environment variables to point at your preferred accelerator (e.g., `CUDA_VISIBLE_DEVICES`).

## Training with a single Reasoning Gym task

```bash
python train.py \
  --task basic_arithmetic \
  --segments 3 \
  --train_n 2000 \
  --eval_n 256 \
  --batch_size 16
```

Adjust the numeric knobs to scale dataset size, HRM depth, or evaluation cadence.

## Mixing multiple tasks

To interleave examples from several Reasoning Gym datasets, use the `--tasks` flag with a comma-separated list:

```bash
python train.py \
  --tasks basic_arithmetic,gsm_symbolic,chain_sum,simple_equations \
  --segments 3 \
  --train_n 4000 \
  --eval_n 512 \
  --batch_size 16
```

The wrapper balances each sub-dataset and exposes the originating task in `item["metadata"]["task"]` for logging.

## Evaluation-only runs

You can resume from a checkpoint directory and run evaluation spot checks without additional training steps:

```bash
python train.py \
  --task basic_arithmetic \
  --resume /path/to/checkpoint \
  --eval_only \
  --eval_n 1024 \
  --eval_every 1000
```

## Additional tips

* `--sentinel` appends a stopping token to every gold target so the model learns to terminate cleanly.
* `--system_header` customizes the instruction prefix injected into each prompt.
* `--log_csv` records evaluation curves for later analysis.
* `--train_injector_scale` and `--eval_injector_scale` control the HRM influence during training and evaluation respectively.

Refer to `train.py` and `reasoning_gym_wrapper.py` for the full set of configurable options and implementation details.
