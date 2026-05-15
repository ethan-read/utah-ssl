# POSSM Reproduction Results

This note tracks the current POSSM-style reconstruction plus phoneme CTC fine-tuning experiments on `brain2text24`.

Hyperparameters are intentionally summarized at a high level because the notebook is still being tuned.

## Current Setup

- dataset: `brain2text24`
- feature mode in current runs: `tx_only`
- data mode: normalized
- boundary key mode: session-level
- stage 1: POSSM-style neural reconstruction pretraining
- stage 2: POSSM encoder plus GRU plus post-GRU strided-conv phoneme decoder trained with CTC
- evaluation: Willett-style `competition_train -> competition_test` within-session block split

The earlier flat / suspicious stage-1 reconstruction loss behavior was traced to a data smoothing / stats mismatch. After fixing that, the reconstruction loss curve looked more normal.

## Temporal Patching / Emission Head Clarification

The current POSSM stage-2 decoder is not doing Willett-style pre-GRU temporal patching.

Willett's reference GRU stack first extracts neural feature patches before the recurrent model:

`raw neural bins -> 14-bin patches every 4 bins -> GRU -> phoneme logits`

With `20 ms` bins, this means each GRU input step directly sees about `280 ms` of neural features, and the recurrent model advances every `80 ms`.

The current POSSM implementation instead keeps the POSSM encoder and GRU at the original bin resolution:

`raw neural bins -> POSSM per-bin encoder -> one 256-d vector per bin -> GRU over every 20 ms bin -> causal Conv1d over GRU states -> phoneme logits`

So the `kernel_size=14`, `stride=4` settings in the POSSM notebook refer to the post-GRU causal convolutional emission head, not a pre-GRU patch extractor. Each CTC logit frame is produced from a causal window over roughly `14` GRU states, or about `280 ms` of already-contextualized recurrent states, and logits are emitted every `4` bins, or about every `80 ms`.

This matches the POSSM speech interpretation in the paper notes: set the input-side window length / stride effectively to `1`, let the GRU process every bin, and use a strided output convolution to control CTC emission frequency. It is POSSM-like, but it is not a literal reproduction of Willett's temporal patching front end.

## Session Adaptation Clarification

The POSSM paper mentions session-related adaptation several times in the general architecture and transfer sections. The clearest mechanisms are:

- learned unit embeddings
- learned session embeddings, especially in the generic output/readout query design
- unit identification (`UI`), where new unit embeddings plus session embeddings are trained for unseen recordings while most model weights are frozen
- full finetuning (`FT`) after this lightweight adaptation phase

However, the speech-specific sections in the current paper notes do not clearly state that the speech POSSM-GRU model used the same session-embedding readout mechanism. In fact, the speech setup replaces the generic output cross-attention readout with a `1D` strided convolutional emission head, so it is not obvious where a generic POSSM session embedding would enter the speech decoder.

The current local implementation also has an optional session input adapter:

`x_i -> gamma_session,i * x_i + beta_session,i -> POSSM encoder`

This per-session per-feature affine starts as an exact identity map with `gamma=1` and `beta=0`. It is meant to handle residual session mean/gain drift without changing the pretrained encoder's input distribution at initialization.

An earlier local version used a Willett/Card-inspired `Linear -> Softsign` adapter before the POSSM encoder. That was removed from the active implementation because the data are already z-scored, and `Softsign` squashes normalized values immediately at checkpoint handoff. For example, an input of `2.0` becomes about `0.67`, so the stage-2 encoder no longer sees the same value scale learned during stage-1 reconstruction pretraining.

The current affine adapter is still not clearly described as part of the POSSM speech recipe. It should therefore be treated as a configurable ablation rather than a default assumption of paper-faithful reproduction. A clean comparison should include at least:

- adapter off: closest to direct stage-1 checkpoint handoff into POSSM-GRU CTC fine-tuning
- affine adapter on: true-identity per-session feature gain/offset adaptation before the POSSM encoder
- possible future variant: POSSM-style session embedding integrated into the speech emission head, if we decide on a principled design for where it belongs

## Current Results

### Pretrained POSSM + GRU

Approximate current result:

- phoneme error rate: about `0.58`
- decoded prediction length: about `76%` of target length
- blank frame rate: about `0.59`

Prediction distribution looked broad enough to not be a simple collapse. The most common predicted phonemes overlapped substantially with the most common target phonemes.

Most common predicted phoneme IDs from the inspected run:

- `40`, `31`, `3`, `6`, `17`, `23`, `20`, `18`, `34`, `22`

Most common target phoneme IDs from the same inspection:

- `40`, `31`, `3`, `23`, `17`, `28`, `29`, `9`, `18`, `21`

Interpretation:

- the model is learning a real but still weak decoder
- it is not just emitting one common phoneme
- it remains deletion-heavy / under-emissive
- common phonemes such as `SIL`, `T`, `AH`, and `IH` appear to be learned better than less frequent or harder classes

### Random-Init POSSM + GRU Baseline

Approximate current result:

- phoneme error rate: about `0.71`
- decoded prediction length: about `57%` of target length
- blank frame rate: about `0.79`

The prediction distribution was strongly collapsed:

- predicted `40`: `1538`
- predicted `3`: `640`
- predicted `6`: `160`

This is qualitatively different from the pretrained run. The random-init baseline mostly emitted a small set of high-prior classes, especially `SIL`.

### Reduced Post-GRU Conv-Stride Stage-2 Run

A later pretrained stage-2 run tested a smaller post-GRU convolutional emission stride together with other stage-2 hyperparameter changes.

This did not improve decoding.

Validation reports across the run were noisy:

- best observed phoneme error rate: about `0.64`
- final phoneme error rate: about `0.71`
- validation CTC stayed around the mid `3` bits/phoneme range

Decoded outputs were also substantially shorter than the previous pretrained run:

- mean decoded prediction length: about `11.5` phonemes
- mean target length: about `26.1` phonemes
- mean prediction / target length ratio: about `0.44`

Interpretation:

- reducing the post-GRU emission stride did not fix under-emission in this run
- the model actually became more deletion-heavy
- the previous pretrained run with the earlier stage-2 settings remains the stronger reference point so far

### 2026-05-12 Lower-Dropout / 3000-Step Stage-2 Run

The latest inspected run returned to the `kernel_size=14`, `stride=4` post-GRU emission head, kept the session adapter off, and reduced GRU dropout relative to the previous strong run.

Run:

- notebook: `analysis/active/ssl_experiments/s6_possm_maskedreconstruction.ipynb`
- stage-1 checkpoint: `possm_stage1_tx_only_normalized_20260512T101159Z/checkpoint_best.pt`
- stage-2 run: `possm_stage2_finetune_full_tx_only_20260512T143912Z`
- mode: `finetune_full`
- feature/data/boundary: `tx_only`, normalized, session boundary keys
- stage-2 cache: raw `cache_v1`, with online Gaussian smoothing after augmentation
- session adapter: off
- steps: `3000`
- validation cadence: every `100` steps
- batch size: `32`
- decoder learning rate: `2e-4`
- encoder learning rate: `3e-5`
- weight decay: `1e-3`
- online smoothing: `sigma_bins=2.0`, `kernel_size=100`, `threshold=0.01`
- training augmentations: `white_noise_sd=0.1`, `constant_offset_sd=0.05`
- decoder: `5`-layer GRU, hidden size `768`, GRU dropout `0.2`
- post-GRU emission conv: `kernel_size=14`, `stride=4`, dropout `0.1`

Validation trajectory:

- step `100`: PER `1.000`, prediction/reference token ratio `0.000`, blank frame rate `1.000`
- step `1000`: PER `0.664`, ratio `0.526`, blank `0.747`
- step `1600`: PER `0.607`, ratio `0.789`, blank `0.629`
- step `2000`: PER `0.575`, ratio `0.762`, blank `0.610`
- step `2500`: PER `0.548`, ratio `0.861`, blank `0.564`
- step `2700`: PER `0.539`, ratio `0.831`, blank `0.599`
- step `2900`: validation CTC `2.795` bits/phoneme, PER `0.529`, ratio `0.808`, blank `0.607`
- step `3000`: validation CTC `2.837` bits/phoneme, PER `0.523`, ratio `0.808`, blank `0.566`

The best validation CTC was at step `2900`, but the best observed PER was the final step `3000`. This is the strongest POSSM stage-2 PER observed so far in these notes.

Prediction diagnostics on `4` validation batches from the saved best-checkpoint path:

- mean decoded prediction length: `21.3` phonemes
- mean target length: `26.1` phonemes
- mean prediction / target length ratio: `0.820`
- median prediction / target length ratio: `0.802`
- diagnostic blank frame rate: `0.566`
- mean logit frames: `72.1`

Most common predicted phoneme IDs in the diagnostic sample:

- `40`, `3`, `31`, `17`, `10`, `23`, `21`, `6`, `22`, `2`

Most common target phoneme IDs in the same sample:

- `40`, `31`, `3`, `23`, `17`, `28`, `29`, `9`, `18`, `21`

Interpretation:

- reducing GRU dropout from `0.3` to `0.2` while training for `3000` steps improved PER and kept length calibration roughly in the good range
- the model is still deletion-biased, but the collapse is much less severe than the reduced-stride run
- full-validation prediction/reference ratio stayed near `0.81` late in training, while blank rate fell into the mid/high `0.5` range
- validation CTC and PER are not perfectly aligned; checkpoint selection by CTC may not always select the best PER checkpoint
- the next comparison should separate the effects of longer training and lower dropout, since both changed relative to the earlier `2000`-step, `gru_dropout=0.3` run

Follow-up continuation:

- a warm continuation from the step-`3000` final checkpoint to step `4000` produced a slightly better PER checkpoint/final result, around `0.517`
- the improvement was small relative to the step-`3000` PER of `0.523`
- validation CTC did not improve; it moved from `2.837` at step `3000` to about `2.99` at step `4000`, with intermediate continued checkpoints mostly in the high `2.8` to low `3.0` range
- train CTC continued to fall strongly during the extension, from roughly `2.27` near step `3000` to roughly `1.54` at step `4000`
- this looks like emerging overfitting: the extra training can occasionally find a slightly lower PER point, but the average continued checkpoint is not clearly better than the pre-extension run, and the train/validation CTC gap widens

## Current Interpretation

The reconstruction-pretrained POSSM initialization appears to help CTC fine-tuning materially:

- lower PER than random initialization
- less blank-frame dominance
- longer decoded outputs
- broader predicted phoneme distribution

This suggests the reproduction is now on a plausible path. The result is not yet useful decoding, but it is no longer the earlier degenerate common-phoneme collapse.

## Willett Smoothing Comparison

Willett's `speechBCI` decoder applies Gaussian smoothing inside the training input transform, after normalization and after training-time input augmentations such as white noise and constant offsets. The local reference code lives at `external/speechBCI`; the relevant functions are `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py::gaussSmooth` and `_datasetLayerTransform`.

Earlier POSSM reconstruction runs used a pre-smoothed cache (`cache_v1_smoothed_sigma2p0`) and kept runtime smoothing disabled. The cache is built by `analysis/active/ssl_experiments/build_smoothed_cache.py`, which calls `masked_ssl.cache._apply_gaussian_smoothing`. That implementation applies an analytic sigma-bin Gaussian with reflect padding to feature arrays before training and records smoothing provenance in dataset metadata.

Important nuance: Willett smooths after adding noise/offset. If POSSM uses a pre-smoothed cache, then any future training-time white-noise or constant-offset augmentation would not be smoothed unless we explicitly smooth after augmentation. That is a small but real difference from the Willett training recipe.

Stage 2 has therefore been moved toward the cleaner Willett-style option: load the raw cache for phoneme fine-tuning, normalize raw target-session features, add training-time white noise / constant offsets, and then apply online Gaussian smoothing before the POSSM encoder. Stage 1 can still use the pre-smoothed cache for reconstruction pretraining.

## Next Things To Check

- aligned error breakdown:
  - insertions
  - deletions
  - substitutions
  - per-phoneme recall
- stage-2 learning curves:
  - train CTC
  - validation CTC
  - validation PER
- CTC emission frequency:
  - reducing the post-GRU conv stride was not automatically helpful
  - future stride changes should be treated as controlled ablations, not assumed improvements
  - compare PER together with prediction / target length, deletion rate, and insertion rate
- rerun pretrained and random-init baselines under the same stage-2 settings whenever changing major decoder hyperparameters
