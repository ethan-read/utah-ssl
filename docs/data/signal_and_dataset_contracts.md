# Signal and dataset contracts

The current code separates three decisions that had previously become mixed
across cache, notebook, and model configuration:

- `DatasetPlan` names every dataset and allowed source split.
- `SignalSpec` defines the modalities, channel ranges, dimensions, and missing
  channel policy consumed by an analysis or model.
- `ExperimentRecipe` gives a scientific question a stable name by pairing one
  dataset plan with one signal specification.

Every call still carries an explicit modality rather than relying on a hidden
software default. Scientifically, however, Brain-to-Text work prefers SBP and
always selects 128 area-6v channels; broad auxiliary-dataset work uses TX when
SBP is unavailable. Brain-to-Text SBP is read from clipped-FP16 caches. Training
configurations additionally carry the exact `DatasetPlan`. Checkpoints and
normalization artifacts serialize both contracts, so a run cannot silently
change its input data when it resumes.

## Named POSSM recipes

`experiments/possm_style/recipes.py` is the authoritative list:

- `POSSM_B2T24_SBP`: Brain2Text24 `competition_train`, 128 area-6v SBP.
- `POSSM_B2T24_B2T25_SBP`: Brain2Text24 `competition_train` plus
  Brain2Text25 `train` and `val`, 128 area-6v SBP.
- `POSSM_BROAD_TX`: the named heterogeneous BIT dataset plan with a
  256-channel model input. Brain-to-Text contributes only its 128 area-6v TX
  channels; narrower physical inputs are zero-padded by the signal contract.
  Every source split is enumerated in `BIT_STAGE1_DATASET_SPLITS`;
  Brain2Text24 `competition_test` is not part of the plan.

`tx_only` and `tx_sbp` remain available for explicit modality comparisons, but
they are not implicit Brain-to-Text defaults. SBP is preferred for
speech-focused work because it has been more reliable than TX in these data.
TX is used for broad heterogeneous pretraining because several datasets have no
SBP.

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

Reusable code follows the same boundary:

- `utah_ssl.canonical_data` owns manifest records and memory-mapped shards;
- `utah_ssl.dataset_splits` builds labeled train/validation problems;
- `utah_ssl.feature_stats` computes and applies normalization;
- `utah_ssl.sequence_data` owns PyTorch datasets, collation, and batching;
- `utah_ssl.cache` prepares multi-dataset cache contexts for segment sampling.

## Loading and normalization guarantees

- Cache access opens only the modalities requested by `SignalSpec`.
- A requested modality or channel range that is absent raises an error unless
  the selected recipe explicitly uses zero-padding.
- Session and split normalization artifacts must match the complete
  `DatasetPlan`, `SignalSpec`, cache identity, and split policy.
- Raw and pre-smoothed caches require their corresponding normalization
  artifacts; smoothing must not be applied a second time at training.

## Checkpoint and resume semantics

Current Stage-1 recovery checkpoints retain model and optimizer state, training
progress, sampler state, masked-objective state, and Python, NumPy, CPU Torch,
and CUDA Torch random-number state. Resume Stage 1 from `checkpoint_final.pt`
or a periodic file under `checkpoints/`; `checkpoint_best.pt` is an evaluation
artifact and is not a valid exact-resume source.

Current Stage-2 recovery checkpoints retain model and optimizer state, progress,
Torch random-number state, deterministic sampler iteration, and the exact
microbatch cursor. This preserves dropout, augmentation, and data-order state
across interruption.

Resume only when the dataset, signal, architecture, and existing training
budget remain compatible with the checkpoint. To use an encoder with a
different downstream signal—such as transferring broad TX pretraining to SBP—
the workflow must implement and document an explicit signal handoff rather
than pretending the checkpoint contracts are identical.

## Failure policy

Current artifacts without `dataset_plan` or `signal_spec` are intentionally
rejected. The error is fixed by regenerating the artifact or starting a fresh
run with an explicit recipe. This keeps recovery code small and makes the
scientific boundary visible instead of guessing how an old configuration
should be interpreted.
