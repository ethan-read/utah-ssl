# SSL Experiments

This folder contains active Utah-array SSL experiments for speech decoding.
The current direction is generic SSL pretraining for `S5` and `Mamba`
encoders, evaluated by downstream Brain2Text24 phoneme CTC decoding.

## Current Packages

- `ssl_core/`: shared cache, stat, CTC, patching, import-path, and reporting
  helpers. New experiment code should import reusable pieces from here.
- `ssm_ssl/`: active generic SSM SSL path. Use this for masked neural modeling
  with `S5`/`Mamba`, raw-bin/temporal-patch/causal-conv-stem inputs, and
  pretrained-vs-random downstream CTC controls.
- `willett_reconstruction/`: supervised Willett-style CTC baseline and recipe
  reference.
- `possm_ssl/`: POSSM implementation retained as a reference and evidence
  source.
- `masked_ssl/` and `contrastive_ssl/`: legacy SSL paths kept for compatibility
  while useful pieces are migrated into `ssl_core` and `ssm_ssl`.

## Preferred Workflow

Start with `ssm_ssl/scripts/run_generic_ssm_ssl.py` for new generic SSL
experiments. It writes stdout progress plus `progress.jsonl`, `metrics.csv`,
checkpoints, and `summary.jsonl`.

Use `willett_reconstruction/` when checking supervised recipe details,
normalization, smoothing, temporal patching, or decoder diagnostics. Check
`docs/notes/cache_and_stats_inventory.md` before changing cache or
normalization-stat paths.

Older exploratory notebooks live in `archive/notebooks/` once their conclusions
have been recorded in `docs/notes/experiment_synthesis.md` or archived notes.
