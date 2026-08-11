# ssl_autoresearch

This directory is the full SSL autoresearch benchmark surface.

The benchmark contract is now active, and the current runnable reference model family is:

- `backbone = s5`
- `objective_family = future_prediction`
- downstream metric = held-out-session `session_avg_val_bpphone`

## Read First

Before changing anything, read:

- `README.md`
- `AGENTS.md`
- `prepare.py`
- `data.py`
- `search_space.py`
- `train.py`
- `run_experiment.py`

## Current Intent

This folder is the main benchmark surface for the current benchmark loop. Use it to:

- validate the held-out-session phoneme benchmark
- compare adaptation regimes `A`, `B1`, and `B2`
- extend the `s5` reference path with `Mamba` and broader SSL objectives without changing the benchmark contract

## Planned Benchmark Regimes

The intended downstream comparison is currently:

- `A`: target-day normalization only + phoneme head
- `B1`: target-day normalization + affine-only input adaptation + phoneme head
- `B2`: target-day normalization + full-encoder fine-tuning + phoneme head

Interpretation:

- the phoneme head is always trained
- the regime controls what additional encoder-side adaptation is allowed
- `B2` is a secondary comparison, not the main inner-loop benchmark

The current primary metric is:

- `session_avg_val_bpphone`

defined as session-averaged validation bits per target phoneme on held-out sessions.

## Editable Surface

You may modify:

- `train.py`

You should avoid modifying during experiment loops:

- `prepare.py`
- `data.py`
- `search_space.py`
- `run_experiment.py`

unless the benchmark contract itself is being intentionally revised.

## Current Search Region

The current runnable search region is:

- `backbone in {s5}`
- `objective_family in {future_prediction}`
- `adaptation_regime in {A, B1, B2}`
- `patch_size in {1, 3, 5}`
- `patch_stride in {1, 3, 5}` with `patch_stride <= patch_size`
- `standardize_scope in {subject, session}`
- `post_proj_norm in {none, rms}`

`debug_gru` is only the bootstrap fallback encoder. Expanding beyond `s5` to `Mamba` should happen by implementing it inside `train.py`, not by changing the benchmark harness.

## Current Limitation

Do not treat the current `s5 + future_prediction` reference path as the final thesis model family.

The metric and loop are real, but the objective family and backbone comparison set are still intentionally narrow so the benchmark contract stays auditable while the fuller model family is added.

## Dry-Run Example

```bash
python run_experiment.py \
  --profile single_gpu \
  --backbone s5 \
  --objective-family future_prediction \
  --adaptation-regime A \
  --patch-size 1 \
  --patch-stride 1 \
  --standardize-scope subject \
  --post-proj-norm rms \
  --dry-run \
  --note "platform check"
```

## Next Implementation Step

The most likely next step is to extend the current in-scope model family:

- implement `Mamba`
- widen the objective family beyond `future_prediction`
- keep the same held-out-session phoneme benchmark and adaptation regimes
