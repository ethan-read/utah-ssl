# S5 Masked Reconstruction Note

This note records the current status of the `S5` causal masked-reconstruction SSL line.

It is intended to capture what has been implemented, what has already been tried, and what the observed results are.

Possible future architecture directions are intentionally deferred to a separate follow-up document.

## Motivation

- contrastive SSL has not yet been clearly helpful for downstream phoneme decoding
- a masked-reconstruction objective seemed worth trying as a more direct signal-modeling alternative
- the goal was to keep the setup causal and `S5`-based rather than switching to a bidirectional encoder

## Latest Run Snapshot (Hyperparams + Result)

- objective: masked reconstruction (`loss_mode="masked_only"`, `reconstruct_target="raw_patch"`)
- backbone: `S5`, `backbone_direction="bidirectional"`, `num_layers=4`, `hidden_size=256`, `s5_state_size=128`
- patching: `patch_size=4`, `patch_stride=2`
- masking:
  - `mask_unit="patch"`
  - `mask_ratio=0.15`
  - `span_length_min=2`
  - `span_length_max=4`
  - `num_spans_mode="multiple"`
  - `mask_token_placement="before_projection"`
- decoder head: `reconstruction_head_type="mlp"`
- optimization: `batch_size=32`, `learning_rate=1e-4`, `weight_decay=0`, `dropout=0`
- preprocessing:
  - `use_normalization=true`
  - gaussian smoothing enabled with `gaussian_smoothing_sigma_bins=2.0`
  - run used precomputed smoothed session stats from Drive
- result (single run):
  - at `step=5000`, train loss reached `0.47` and had not plateaued yet
  - validation loss on a different session reached `0.70`

## Medium-Scale Brain2Text24 Holdout Run (Current)

- scale / split:
  - dataset: `brain2text24` only
  - chronological subset: latest `12` sessions
  - SSL split: `9` train sessions (`2022-07-05` through `2022-08-13`) and `3` val sessions (`2022-08-18`, `2022-08-23`, `2022-08-25`)
- SSL hyperparameters:
  - target run length: `10000` steps
  - `mask_ratio=0.25`
  - gaussian smoothing: `gaussian_smoothing_sigma_bins=2.5` with precomputed session stats
  - logging / eval / checkpoint cadence: `log_every=25`, `val_every=100`, `checkpoint_every_steps=1000`
  - architecture and optimizer otherwise unchanged from the active `S5` masked-reconstruction config (`hidden_size=256`, `num_layers=4`, `patch_size=4`, `patch_stride=2`, `learning_rate=1e-4`, `batch_size=32`)
- downstream probe readout (current checkpoint, short probe):
  - probe setup: frozen encoder, linear probe, `80` probe steps, evaluated on held-out val set
  - result:

| model_variant | downstream_ctc_bpphone | downstream_per | reference_output_len | actual_output_len | actual_over_reference_len | most_common_prediction | most_common_prediction_rate |
|---|---:|---:|---:|---:|---:|---|---:|
| `ssl_run_probe80_colab_s5_masked_reconstruction...` | 9.146503 | 0.958631 | 701 | 744 | 1.061341 | SIL | 0.520161 |

Brief takeaway (latest reruns): this larger masked-reconstruction run on most of `brain2text24` produced only minimal downstream gains with the current hyperparameters.

## Sweep Update (Smoothing Sigma)

- a small sweep over `gaussian_smoothing_sigma_bins in {1.0, 2.0}` and `mask_ratio in {0.15, 0.25}` was run (`1500` SSL steps + short probe)
- tentative read:
  - `sigma=2.0` looked better for SSL reconstruction loss (lower train/val loss than `sigma=1.0` in matched settings)
  - `sigma=2.0` also produced longer probe outputs (`output_len_over_desired_len` higher), i.e. less under-generation
- downstream CTC/PER differences were small across these short probes, so this should be treated as directional rather than final

## Zero-Predictor Baseline Update

- on the currently smoothed and normalized data, the constant-zero predictor MSE appears to be `0.95+` for the active sessions
- this means the trivial baseline is still close to `1.0`, but future reconstruction losses should be compared to the measured session/mask-specific zero baseline rather than assuming exactly `1.0`
- a model loss around `0.75` is therefore a real improvement over the zero predictor, but only a modest one; downstream probe behavior remains the key test of whether the representation is useful

## What Has Been Implemented

- a new reusable package at `analysis/active/ssl_experiments/masked_ssl`
- a new Colab notebook at `analysis/active/ssl_experiments/s5_maskedreconstruction.ipynb`
- a causal `S5` encoder with a reconstruction head that predicts raw patched neural signal values
- support for:
  - patch masking
  - bin masking
  - mask-token insertion before or after the input projection
  - masked-only reconstruction loss
  - optional subject-aware boundary routing for session-specific read-in/read-out layers
  - downstream `Brain2Text25` frozen phoneme probing
- tests for:
  - masking behavior
  - checkpoint recovery
  - downstream probe loading
  - causal-prefix invariance
  - subject-aware boundary-key routing

## Initial Experiment Design

The first version of the experiment used:

- `patch_size = 5`
- `patch_stride = 5`
- `mask_unit = "patch"`
- `mask_ratio = 0.40`
- `span_length_min = 2`
- `span_length_max = 8`
- `num_spans_mode = "one"`
- loss on all masked patches

The objective was:

- replace masked patches with a learned mask token
- run the masked sequence through a causal `S5`
- reconstruct the original normalized patch values
- compute MSE only on masked elements

## Boundary Routing

The masked-reconstruction code now distinguishes two levels of adaptation:

- the core causal `S5` backbone remains shared
- the boundary affine layers can be routed by either session or subject

The current implementation adds a small `boundary_key_mode` switch:

- `session`: keep the original session-specific routing
- `subject_if_available`: use `subject_id` when it exists in the manifest, otherwise fall back to `session_id`

This is useful for the `motor_data` and `brain2text25` cache layouts, where `subject_id` is already present in the manifest rows.

The intent is to keep the code simple while making the read-in / read-out layers less fragmented across many sessions from the same subject.

## Diagnostic Additions

To understand failure modes better, the following diagnostics were added:

- masked prediction mean
- masked prediction standard deviation
- masked target mean
- masked target standard deviation
- explicit causal-prefix test:
  - two inputs that match on a prefix and differ only later must produce identical hidden states / reconstructions on the shared prefix

## Main Result So Far

The core failure mode has been very consistent:

- training loss quickly falls to about `1.0`
- masked-token MSE also stays around `1.0`
- masked prediction mean stays near `0`
- masked prediction standard deviation stays near `0`

This happened in the original setup and remained true after a first round of simplifications.

## Interpretation

This strongly suggests collapse to the trivial normalized-mean predictor.

Why `1.0` matters:

- the reconstruction targets are session-featurewise z-scored
- if the model predicts a constant `0` on masked elements, expected MSE is about the variance of the normalized target
- since the normalized target variance is about `1`, the trivial `predict 0` solution gives loss about `1`

The observed combination:

- target mean near `0`
- target std near `1`
- prediction mean near `0`
- prediction std near `0`

is exactly the signature of that failure mode.

## First Round Of Fixes Already Applied

The experiment was then simplified in a more causal-friendly direction:

- `patch_size` changed from `5` to `4`
- `patch_stride` changed from `5` to `2`
- `mask_ratio` reduced to `0.15`
- `span_length_min = 1`
- `span_length_max = 1`
- `num_spans_mode = "multiple"`
- for longer patch spans, reconstruction loss was changed to score only the first masked patch in each span

The reasoning was:

- long fully masked contiguous spans are a poor fit for a causal model
- once the model is deep inside a masked span, it no longer has real local input evidence
- the hope was that shorter, lighter masking would let the model leave the `predict 0` regime

## Current Update

Even after those changes, the loss is still reported to be stuck around `1.0`.

So at the moment the masked-reconstruction line should be treated as:

- implemented
- instrumented
- diagnostically clearer than before
- but still empirically negative

## Current Conclusion

- the present causal masked-reconstruction setup is not yet learning a useful nontrivial predictor
- more data alone should not be expected to fix this if the model is still sitting at the normalized-mean baseline
- the immediate bottleneck appears to be objective / supervision design rather than simple data scale

## Scope Boundary For This Note

This document is intentionally limited to:

- implementation status
- experiment settings tried so far
- observed behavior
- current empirical conclusion

It does not yet propose the next architecture directions in detail.

Those should go in a separate follow-up note.
