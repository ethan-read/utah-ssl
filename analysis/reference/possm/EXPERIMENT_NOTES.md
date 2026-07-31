# POSSM experiment handoff notes

This is the compact handoff for the current POSSM reconstruction and transfer
experiments. The implementation lives in this folder; the reusable cache and
normalization artifacts live under the Drive data root, not in experiment
output directories.

## Current question

Test whether adding Brain2Text25 improves transfer to Brain2Text24:

1. Stage 1 pretraining uses Brain2Text24 `competition_train` plus Brain2Text25
   `train` and `val`.
2. Brain2Text24 `competition_test` and Brain2Text25 `test` are excluded from
   Stage 1. Brain2Text25 `val` is used as unlabeled reconstruction data only;
   its labels are not used.
3. Stage 2 is unchanged: fine-tune on Brain2Text24 `competition_train` and
   evaluate on Brain2Text24 `competition_test`.

This is a pooled, mildly transductive pretraining experiment with respect to
the Brain2Text25 validation distribution, but the final held-out metric is
still Brain2Text24 test PER. The comparison should use the same Stage-2 recipe
and 12,000-step budget as the Brain2Text24-only baseline.

## Where to work

- `notebooks/s6_possm_maskedreconstruction.ipynb`: historical Brain2Text24-only
  baseline record. It preserves the original comparison but is not maintained
  as a current entry point after the explicit contract migration.
- `notebooks/s13_brain2text25_long_pretraining.ipynb`: older Brain2Text25-only
  workflow. It contains strict 128-wide cache assertions and is not the clean
  entry point for the pooled experiment.
- `notebooks/s14_possm_pooled_pretraining.ipynb`: current Colab workflow.
  Edit its single configuration cell, then run the cells in order. It is
  designed for Drive-backed Colab runs; no local training smoke test is
  expected.
- `notebooks/s15_possm_pooled_cache_preparation.ipynb`: one-time workflow that
  builds and validates the optimized versioned raw/smoothed cache roots,
  recomputes pooled stats, and benchmarks cold versus warm sampling.
- `possm_ssl/model.py`: POSSM encoder and phoneme model architecture.
- `possm_ssl/training.py`: Stage-1 training, checkpoints, recovery, and
  progress serialization.
- `possm_ssl/phoneme_finetune.py`: Stage-2 data loading, fine-tuning,
  checkpoint recovery, CTC loss, PER, and timing logs.
- `SIGNAL_AND_DATA_CONTRACTS.md`: the current modular interface for choosing
  datasets and neural signals across cache inspection, analysis, and training.
- `../active/ssl_experiments/masked_ssl/cache.py`: shared cache context and
  dataset-specific source-split filtering.
- `../active/ssl_experiments/ssl_core/scripts/recompute_feature_stats.py`:
  model-independent session/global normalization-statistics generator.
- `docs/notes/cache_and_stats_inventory.md`: canonical cache roots and stats
  inventory.
- `docs/notes/archive/possm_reproduction_results.md`: detailed history of the
  earlier Brain2Text24 reconstruction/fine-tuning runs, including the 12k-step
  result and the evidence that the cached inputs were already z-scored.
- `docs/notes/experiment_synthesis.md`: compact cross-experiment interpretation
  of POSSM relative to the supervised Willett-style baselines.

The pooled notebook uses one explicit `DatasetPlan`:
`brain2text24=(competition_train,)` and
`brain2text25=(train,val)`. Dataset selection is positive-only: the plan names
every dataset and split that participates, and the cache layer has no
“everything except these datasets” configuration.

## Colab recovery and outputs

Stage 1 has a fixed run name and a separate `pooled_pretraining` output root.
Use `STAGE1_STATE_MODE='train'` only for a new run. If Colab stops, change it
to `resume` (or recover a specific checkpoint) and rerun the setup and Stage-1
cells. The target remains `STAGE1_RESUME_TARGET_STEPS=12000`.

Stage 2 writes to the separate `pooled_pretraining_stage2` root. Use
`STAGE2_RUN_ACTION='fresh'` for the first fine-tune, and
`'resume_latest'` or an explicit recovery path after interruption. A Stage-2
checkpoint contains the model/optimizer, Torch RNG, deterministic sampler
iteration, exact microbatch cursor, config, and progress state needed to
continue. New checkpoints therefore reproduce the uninterrupted next-step
trajectory, including noise augmentation and dropout. Older checkpoints remain
resumable but print a warning and restart the deterministic data order because
they do not contain the exact cursor. Keep the dataset, architecture, and step
budget unchanged when resuming.

The shared refactor and migration behavior are documented in
`docs/notes/ssl_signal_contract_refactor.md`.

The notebook records Stage-2 train CTC, validation CTC, validation PER,
checkpoint step, and `sample_seconds`/`model_seconds` in the progress log.
Because validation CTC begins to plateau or rise while PER can continue to
fall, inspect both curves and retain the best-PER checkpoint for the main
transfer result, while reporting the best validation-CTC point as a secondary
diagnostic.

## Cache and stats pitfalls

Session statistics are a one-time CPU/Drive scan over every selected
pretraining example. The pooled artifact is stored under the organized
`utah_ssl/data/stats/session_feature_stats/...` hierarchy, with a filename
identifying the selected Brain2Text24/25 splits. Reuse it on later notebook
runs. If `prepare_cache_context` reports that the artifact is stale or
incompatible, run the exact `recompute_command` printed in the exception or by
the notebook; do not bypass the metadata check.

The canonical Brain2Text25 cache may be 256-wide. The pooled workflow reads
Brain2Text25 from the optimized 128-wide root described below, while
Brain2Text24 is read directly from its unchanged canonical root.

The notebook's `torch.load` compatibility cell is intentionally idempotent.
If cells are rerun, keep the saved original loader rather than wrapping the
already-wrapped function again; otherwise a recursive wrapper can produce a
`RecursionError` during checkpoint recovery.

### Optimized pooled cache

Future fresh pooled runs use these Brain2Text25-only versioned roots:

- raw:
  `utah_ssl/data/cache_v1_possm_b2t25_area6v_v1`
- smoothed:
  `utah_ssl/data/cache_v1_possm_b2t25_area6v_sigma2p0_v1`

They contain only Brain2Text25 and are built independently from the existing
canonical raw and smoothed roots. Brain2Text24 is neither copied nor
transformed. Brain2Text25 retains 128 area-6v TX and 128 area-6v SBP channels
and targets approximately 65 MiB per fused shard, matching the median
Brain2Text24 shard size. Retained TX, SBP, labels, and offsets preserve their
source dtypes and values exactly. Building the smoothed destination from the
existing smoothed source is deliberate: smoothing is not regenerated after
examples have been assigned to new shard boundaries.

An earlier pooled run stored optimized Brain2Text25 TX as `float16`. The
current preparation workflow no longer performs this conversion: its only
feature transformation is selecting columns `[0, 128)` for TX and SBP.
Preserving SBP exactly is particularly important for planned SBP-only
cross-year comparisons.

The cache loader is modality-aware. A `tx_only` context opens TX but not SBP;
an `sbp_only` context opens SBP but not TX. The pooled POSSM recipe explicitly
uses 128-channel area-6v SBP. Stage-1 logs include cumulative cache hit rate,
cached GB, bytes read, and evictions. Use `drive_direct` first; switch s14's
`STAGE1_CACHE_MODE` to `copy_to_local` only if the s15 warm benchmark remains
slow.

There is deliberately no global signal default. Each run selects a named
recipe or constructs a `DatasetPlan` and `SignalSpec`; `tx_only`, `sbp_only`,
and `tx_sbp` are all valid explicit choices. Cache loading, statistics, raw
access, manifold/probe construction, and model training consume the same
objects. Current checkpoints and stats artifacts must contain both contracts;
incomplete old artifacts fail with a direct missing-contract error. Stage 2
inherits the exact Stage-1 signal unless an identical `SignalSpec` is supplied.

Both Stage-1's eager shard cache and Stage-2's memory-mapped shard accessor are
modality-aware. In particular, an SBP-only Stage-2 dataset does not open or
memory-map `tx.npy`.
Synthetic tests cover the SBP-only Stage-1 checkpoint handoff, Stage-2
fine-tuning checkpoint creation, and Stage-2 resume.

The mixed-root view has a composite source signature and a model-independent
stats namespace:
`stats/session_feature_stats/smoothed_sigma2p0_mixed_<source-signature>/`.
Do not reuse the old pooled artifact by path. Do not resume a partially trained
run across old and optimized shard topologies; optimized-cache runs start from
step zero under their separate run/output names.

The active S14 workflow should expose only two required normalization
artifacts: pooled pre-smoothed SBP session stats for Stage 1 and raw
Brain2Text24 `competition_train` SBP global stats for Stage 2. Add the
Brain2Text24-only pre-smoothed SBP session artifact only when running a fresh
SBP baseline. TX/TX+SBP and `stage1_global_feature_stats` files are historical
reproduction inputs, not current defaults. Use `ssl_core.stats` from Python and
`recompute_feature_stats.py` from notebooks; do not introduce another
POSSM-specific stats namespace.

## Speedup idea: CUDA BF16 Stage 2

The current Stage-2 step is about 0.3 seconds, with model compute dominating
sampling time. A secondary implementation could use CUDA BF16 autocast for
the forward pass while converting logits to float32 before CTC loss. On an L4
or A100, a reasonable expectation is roughly 0.18--0.24 seconds per step
(about 1.3--1.7x); on a T4 the gain may be small because the packed 5-layer
GRU remains a bottleneck.

BF16 model compute is separate from cache storage. The current optimized cache
preserves source dtypes; this section concerns optional Stage-2 mixed-precision
model computation. BF16 is numerically close to FP32 but not bitwise
identical. Therefore:

- keep the current FP32 workflow as the primary baseline-vs-pooled comparison;
- use BF16 as a speed-optimized secondary run, or rerun both experiments with
  the same BF16 setting if strict precision-controlled comparison is needed;
- preserve the 12,000-step budget and log the actual per-step timing and final
  PER rather than assuming the expected speedup.

S5 is intentionally not part of the current comparison.
