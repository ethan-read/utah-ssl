# Willett Reconstruction Baseline

This package trains a Willett-style supervised GRU phoneme decoder on the
canonical Utah cache and released `competition_train -> competition_test` split.

The first intended comparison target is:

- dataset: `brain2text24`
- feature mode: `tx_only`
- cache root: raw canonical cache
- normalization: block-wise z-scoring by default
- smoothing: Willett-style online Gaussian smoothing

## Local Run

Run from the repo root with:

```bash
PYTHONPATH=analysis/active/ssl_experiments \
python -m willett_reconstruction.train \
  --cache-root /Users/home/thesis/data/cache_v1 \
  --output-root /Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/willett_reconstruction_runs \
  --run-name willett_tx_only_debug \
  --feature-mode tx_only \
  --max-steps 5000 \
  --batch-size 64 \
  --resume-latest
```

## Colab

Use [s7_willett_reconstruction.ipynb](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/s7_willett_reconstruction.ipynb) for:

- Drive mount and repo bootstrap
- cache / stats / output path setup
- script-based training with checkpointing
- loss / PER / blank-rate diagnostics
