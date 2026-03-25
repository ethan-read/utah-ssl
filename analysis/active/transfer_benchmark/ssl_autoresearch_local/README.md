# SSL Autoresearch Local

This directory is a local-only smoke-test scaffold for an autoresearch-style SSL benchmark.

It is intentionally narrower than the intended full CUDA pipeline:

- dataset source: existing Brain2Text25 `TX` and `SBP` caches in `code/ssl/`
- device target: `mps`, then `cpu` fallback, then `cuda` if available
- run budget: fixed `5` minutes per experiment
- optimization target: temporary `val_ssl_loss`

The current local benchmark is meant to validate the automation loop and small-data plumbing on a laptop.

It is not yet the final thesis benchmark.

In particular:

- the metric is currently validation SSL loss, not a downstream transfer score
- `train.py` currently contains a small causal placeholder model to exercise the loop
- the intended `S5` / `Mamba` CUDA benchmark will come later in a separate full pipeline

## Files

- [`prepare.py`](prepare.py): frozen local benchmark contract and constants
- [`data.py`](data.py): fixed data loading for a small Brain2Text25 cache subset
- [`train.py`](train.py): editable training script
- [`run_experiment.py`](run_experiment.py): one-run wrapper and result logger
- [`program.md`](program.md): agent instructions
- [`search_space.py`](search_space.py): compact local smoke search space

## Current Benchmark Contract

- fixed data budget:
  - latest `8` matched Brain2Text25 sessions with both `TX` and `SBP`
  - first `6` selected sessions used for training
  - last `2` selected sessions held out entirely for validation
  - `256` train windows per train session
  - `64` validation windows per held-out validation session
- fixed objective:
  - multi-horizon future prediction in token space
- fixed metric:
  - `val_ssl_loss` on held-out sessions
- fixed run budget:
  - `5` minutes wall clock for training

## Why This Exists

This folder is a staging area before the full CUDA-dependent pipeline.

The goal is to:

1. validate the Karpathy-style experiment loop locally
2. verify MPS compatibility and logging behavior
3. establish a small, reproducible benchmark harness
4. move to a more faithful `S5` / `Mamba` benchmark once the automation loop is working
