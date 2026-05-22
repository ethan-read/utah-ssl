# Willett Reconstruction Replication Notes

## Purpose

Track the status of the local Willett-style supervised phoneme-decoding
baseline in `analysis/active/ssl_experiments/willett_reconstruction`, with a
focus on convergence behavior and remaining discrepancies from the Stanford
`speechBCI` reference code in `external/speechBCI`.

## Current Local Baseline

- data source: canonical Utah cache, not Stanford TFRecords
- split: `competition_train -> competition_test`
- default feature mode: `tx_only`
- model family: pre-GRU temporal patching + GRU + CTC
- session/day adaptation: per-boundary-key `Linear -> Softsign -> Dropout`
  input network applied before patching
- optimizer: Adam with Stanford-style `epsilon=1e-1`
- checkpoint selection: best validation PER

## Confirmed Improvements Already Landed

- session/day input network now matches the Stanford ordering more closely:
  input remapping happens before temporal patching
- adapter keys now use the same boundary-key space as the data loader
- resume checkpoints now preserve `best_step` and `best_progress_payload`

## Important Remaining Discrepancies

### Per-day Sampling Regime

Stanford training still differs in a potentially important way.

The reference code samples one dataset/day at a time during training and routes
that batch through the corresponding day-specific normalization/input path:

- `external/speechBCI/NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py:454`

The current local baseline instead uses one global length-aware sampler over
all utterances:

- `analysis/active/ssl_experiments/willett_reconstruction/train.py:305`

That means a local microbatch can mix multiple sessions/subjects even though
different examples still go through different per-example input networks.

Why this matters:

- it changes the optimization problem seen by the shared GRU stack
- it weakens the original "one day-specific frontend at a time" training regime
- it may slow or destabilize early escape from the CTC blank basin
- it may make adapter learning less clean than the Stanford recipe

Practical interpretation:

- this discrepancy does not prove the local baseline cannot train
- but it is large enough that convergence speed and stability are not yet a
  clean apples-to-apples comparison with Stanford/Willett

### Validation Normalization Leakage

The local code currently computes validation normalization stats from
`val_rows` when using non-global normalization modes. That is not faithful to
the Stanford setup, where normalization is adapted from training data and then
reused at evaluation.

This should be treated as a real bug, not just a recipe difference.

### Missing Trainable GRU Initial State

Stanford's GRU backend uses learned initial recurrent states. The local PyTorch
baseline currently relies on the default zero hidden state for every sequence.

This is a moderate architecture discrepancy that could affect convergence.

### Feature Set Difference

The local baseline still defaults to `tx_only`, whereas Stanford/Willett uses
threshold crossings plus spike-band power together in the standard speech
decoder path.

This is an intentional simplification for comparison against the current POSSM
stage-2 setup, but it still makes the run less faithful to the original.

## Assessment Of The Per-day Sampling Issue

I do think the per-day sampling difference is important enough to document and
keep in mind while interpreting results.

My current view:

- random mixed sampling can still train a usable model
- if the adapters, normalization, and optimizer are all healthy, it does not
  automatically break learning
- however, for this specific architecture family, day-wise batching is probably
  more than a cosmetic detail

Why I think that:

- the Stanford recipe was designed around day-specific normalization and
  day-specific input frontends
- mixing days inside one optimization stream pushes more of the burden onto the
  shared GRU to absorb heterogeneity
- that is especially risky in the early CTC phase, when the model is already
  prone to blank-heavy local optima

So the best summary is:

- `yes`, random sampling can still train fine in principle
- `no`, I would not assume it is equivalent here
- for a faithful replication, per-day sampling should be treated as an
  important remaining discrepancy rather than dismissed as harmless noise

## Suggested Next Fixes

- reuse train-derived normalization stats at validation time instead of
  recomputing them on `val_rows`
- add learned GRU initial states to the PyTorch model
- consider an optional Stanford-style per-day training sampler
- run a direct `tx_sbp` baseline after the `tx_only` path is stable
