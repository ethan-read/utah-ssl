# SSL signal/data contract refactor

This note records the cross-cutting refactor that replaced implicit feature
and dataset defaults with explicit, serializable experiment contracts. It
applies to shared cache access, statistics, SSL training, downstream probes,
manifold/raw-data analysis, and the POSSM reference workflows.

## Why this changed

The previous code represented the scientific data boundary in several
independent settings: feature mode, TX/SBP dimensions, included or excluded
datasets, source-split filters, cache variants, and notebook-local defaults.
Those settings could drift between cache preparation, statistics, Stage 1,
Stage 2, and checkpoint recovery.

The maintained interface now has three objects:

- `SignalSpec`: exact modalities, dimensions, selected column range, and
  missing-channel policy.
- `DatasetPlan`: a positive list of datasets and allowed source splits.
- `ExperimentRecipe`: a named pairing of a signal specification and dataset
  plan for a scientific comparison.

The definitions live in
`analysis/active/ssl_experiments/ssl_core/experiment_contract.py`. Shared
feature-layout behavior lives in `ssl_core/feature_contract.py`. POSSM's named
recipes live in `analysis/reference/possm/possm_ssl/recipes.py`.

There is deliberately no global TX or SBP default. Callers must select a named
recipe or construct an explicit contract. Dataset selection is positive-only;
training code does not infer “all datasets except...” policies.

## Behavioral guarantees

- Cache contexts load only the modalities requested by `SignalSpec`.
- A requested modality missing from a physical shard raises an error.
- With `missing_channel_policy="error"`, a physical array narrower than the
  selected channel range raises an error even if its manifest claims otherwise.
- Zero-padding occurs only when `missing_channel_policy="zero_pad"` is
  explicitly selected.
- Session and split normalization artifacts serialize and validate the complete
  `SignalSpec`, `DatasetPlan`, cache signature, and split/boundary policy.
- Model training configs and checkpoints carry the same signal contract.
- POSSM Stage 2 inherits the Stage-1 signal and rejects a different explicit
  signal.
- Known POSSM single-dataset sweep entry points use leakage-safe source splits:
  Brain2Text24 uses `competition_train`; Brain2Text25 uses `train` and `val`.

## Checkpoint and resume semantics

### POSSM Stage 1

Resumable checkpoints now persist:

- model and optimizer state;
- Python, NumPy, CPU Torch, and CUDA Torch RNG state;
- training and validation sampler RNG state;
- masked-objective generator state;
- current step, histories, best metric, and dataset counts.

This makes a new Stage-1 checkpoint an exact stochastic continuation. A best
checkpoint is an evaluation artifact and is explicitly rejected as a training
resume source. Resume should use `checkpoint_final.pt` or a checkpoint under
`checkpoints/`.

Checkpoints predating RNG/sampler persistence can still be used to initialize
Stage 2, inspect a run, or recover an encoder. They are rejected for further
Stage-1 training because silently restarting their random streams would change
the experiment.

When extending a run's target step count, an earlier best checkpoint remains
compatible if `num_steps` is the only config difference.

### POSSM Stage 2

New resumable checkpoints persist:

- model and optimizer state;
- CPU and CUDA Torch RNG state, covering dropout and neural-data augmentation;
- deterministic length-aware sampler iteration;
- exact number of microbatches already consumed in that iteration.

Stage 2 reconstructs and advances the deterministic DataLoader to the saved
cursor, then restores RNG state. This also handles a checkpoint written after
the final batch of an iteration but before the iterator observes
`StopIteration`.

Older Stage-2 checkpoints remain usable. Resume prints a warning and restarts
from the beginning of the deterministic data order because those checkpoints
do not contain an exact cursor or RNG state.

## Notebook and statistics audit

The maintained pooled POSSM notebook is
`analysis/reference/possm/notebooks/s14_possm_pooled_pretraining.ipynb`.
Its Stage-1 session-stat and Stage-2 split-stat commands now pass the complete
signal contract, including TX/SBP dimensions, column start, missing-channel
policy, boundary mode, and Stage-2 split policy. This prevents a cache's
declared native width from producing a differently shaped artifact at a
128-channel path.

The one-time Brain2Text25 cache preparation remains in
`analysis/reference/possm/notebooks/s15_possm_pooled_cache_preparation.ipynb`.
Cache roots and reusable statistics are inventoried in
`docs/notes/cache_and_stats_inventory.md`.

The historical s6 and s13 POSSM notebooks remain evidence from earlier
experiments; they are not maintained migration targets.

## Validation performed

The refactor audit added regression coverage for:

- requested-modality failures and modality-aware shard I/O;
- strict versus explicit zero-padded channel selection;
- exact Stage-1 sampler state recovery;
- exact interrupted-versus-uninterrupted Stage-2 continuation, including
  augmentation RNG.

Static audit checks additionally covered leakage-safe POSSM single-dataset
plans, notebook JSON/AST validity, and exact notebook statistics arguments.

At the end of the audit, the maintained synthetic/unit suites passed:

- 43 SSL-core tests;
- 53 masked-SSL tests;
- 64 POSSM tests.

Compilation, notebook validation, and `git diff --check` also passed. No
dataset-backed local training run was used; long-running workflow validation
remains a Colab task.

## Migration rule

The code intentionally favors direct, fixable incompatibility errors over
legacy inference. If an artifact lacks `signal_spec` or `dataset_plan`, or its
metadata disagrees with the selected cache, regenerate the artifact or start a
fresh run. Do not add fallback logic that guesses the old scientific boundary.
