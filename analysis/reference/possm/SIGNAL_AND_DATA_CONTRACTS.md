# POSSM signal and dataset contracts

For the broader implementation history, checkpoint semantics, audit fixes, and
cross-workflow guarantees, see
`docs/notes/ssl_signal_contract_refactor.md`.

The current code separates three decisions that had previously become mixed
across cache, notebook, and model configuration:

- `DatasetPlan` names every dataset and allowed source split.
- `SignalSpec` defines the modalities, channel ranges, dimensions, and missing
  channel policy consumed by an analysis or model.
- `ExperimentRecipe` gives a scientific question a stable name by pairing one
  dataset plan with one signal specification.

There is no global modality default and no exclusion-based dataset selection.
Low-level cache, raw-data, statistics, probe, manifold, and model APIs require
an explicit `SignalSpec`. Training configurations additionally carry the
exact `DatasetPlan`. Checkpoints and normalization artifacts serialize both
contracts, so a run cannot silently change its input data when it resumes.

## Named POSSM recipes

`possm_ssl/recipes.py` is the authoritative list:

- `POSSM_B2T24_SBP`: Brain2Text24 `competition_train`, 128 area-6v SBP.
- `POSSM_B2T24_B2T25_SBP`: Brain2Text24 `competition_train` plus
  Brain2Text25 `train` and `val`, 128 area-6v SBP.
- `POSSM_BROAD_TX`: the named heterogeneous BIT dataset plan, 256-channel TX.
  Datasets with narrower native TX are zero-padded by the signal contract.
  Every source split is enumerated in `BIT_STAGE1_DATASET_SPLITS`;
  Brain2Text24 `competition_test` is not part of the plan.

`tx_sbp` remains available for an explicit ablation, but it is not an implicit
fallback. SBP is preferred for the speech-focused cross-year experiment
because the two years have more comparable SBP distributions. TX is preferred
for broad heterogeneous pretraining because several datasets have no SBP.

## One interface across workflows

Raw or manifold analysis should construct a `SignalSpec` and pass it to the
canonical accessor or problem builder. Cache-backed training passes the same
object to `CacheAccessConfig` and the Stage-1 training config. Statistics are
computed for the exact `(DatasetPlan, SignalSpec)` pair.

Stage 2 reads the Stage-1 checkpoint contract and inherits its signal. An
explicit Stage-2 signal is accepted only when it is identical. Changing from
TX to SBP, changing dimensions, or changing selected columns requires a fresh
Stage-1 run and matching statistics.

The cache itself remains modality-neutral. It may contain both TX and SBP;
the accessor opens only the arrays requested by `SignalSpec`. Cache-preparation
scripts may create versioned physical layouts for I/O performance, but they do
not decide which datasets or modalities an experiment uses.

## Failure policy

Current artifacts without `dataset_plan` or `signal_spec` are intentionally
rejected. The error is fixed by regenerating the artifact or starting a fresh
run with an explicit recipe. This keeps recovery code small and makes the
scientific boundary visible instead of guessing how an old configuration
should be interpreted.
