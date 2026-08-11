# POSSM-Style Results

New comparisons should use the shared
[experiment report template](../../../docs/experiment_report_template.md), with
related runs grouped in one report and exact runs recorded as rows. This file
is the canonical summary of completed work.

Unless explicitly stated otherwise, PER values in this summary are the lowest
observed values on the run's validation split. They are not held-out test
estimates, and comparisons across different data mixtures or budgets should be
treated cautiously.

This note is the canonical results record for the paper-derived POSSM-style
reconstruction and Brain2Text24 phoneme-decoding experiments. Implementation
details remain in `../design/implementation_notes.md`.

## Completed TX comparison

The clearest completed historical TX comparison used the area-6v Brain2Text24
setup for 12,000 Stage-2 steps:

- reconstruction-pretrained initialization: validation PER `0.467`;
- matched random initialization: best validation PER approximately `0.707`
  and final validation PER approximately `0.728`.

The random-initialized model was substantially more blank-dominant and
under-emissive. This single-seed comparison supports an optimization benefit
from reconstruction pretraining, but it is not a statistically resolved
effect.

Absolute reconstruction losses from the old TX-only and pooled-TX runs should
not be compared. Their dataset composition, subject, update exposure, and
preprocessing differed; in particular, the historical smoothed Brain2Text24 TX
target was sparse `uint8` while Brain2Text25 TX remained continuous. The pooled
validation aggregate also need not select the best encoder for downstream
Brain2Text24. A shared channel-index embedding across T12 and T15 was a possible
cross-subject confound, not an established explanation.

## 2026-08-01 pooled Brain2Text24/25 SBP result

### Experiment

Stage 1 used:

- Brain2Text24 `competition_train`;
- Brain2Text25 `train` and `val`, with labels ignored;
- 128 area-6v SBP channels;
- the pre-smoothed sigma-2 caches and session-wise normalization;
- plain same-bin reconstruction MSE for 12,000 optimizer steps.

Brain2Text24 `competition_test` and Brain2Text25 `test` were excluded from
pretraining. Brain2Text25 `val` was used only as unlabeled reconstruction data.

Stage 2 used the established compute-matched recipe:

- initialization from the pooled Stage-1 checkpoint;
- Brain2Text24 `competition_train -> competition_test`;
- raw SBP normalized with Brain2Text24 competition-train global statistics;
- training augmentation followed by online sigma-2 Gaussian smoothing;
- full encoder fine-tuning;
- 5-layer, 768-hidden-unit GRU with dropout `0.2`;
- post-decoder causal convolution with kernel `14` and stride `4`;
- session input adapter disabled;
- 12,000 optimizer steps.

Run artifacts:

- Stage 1:
  `sbp_only_pooled_pretraining_b2t25_area6v_v1/possm_stage1_sbp_pooled_b2t24_b2t25_12k_seed7_dataset_metrics_v1`
- Stage 2:
  `sbp_only_pooled_pretraining_b2t25_area6v_v1_stage2/possm_stage2_finetune_full_sbp_only_20260731T172208Z`

### Stage-1 reconstruction

| Measurement | Step | MSE |
|---|---:|---:|
| Best sampled aggregate validation | 11,650 | 0.302185 |
| Final aggregate validation | 12,000 | 0.335834 |
| Final training batch | 12,000 | 0.282970 |

The late training and validation losses remained close enough to look much
healthier than the earlier pooled TX run. Validation used two newly sampled
batches per check, so the minimum should not be interpreted as a stable
full-panel estimate. This run finished before per-dataset MSE logging was
available; the aggregate history cannot be separated retrospectively into
Brain2Text24 and Brain2Text25 curves.

### Stage-2 decoding

| Selection point | Step | Train CTC | Validation CTC | PER | Predicted/reference | Blank rate |
|---|---:|---:|---:|---:|---:|---:|
| Best validation CTC | 9,400 | — | 2.028183 | 0.378979 | 0.883747 | 0.542755 |
| Best observed PER | 11,700 | — | 2.110521 | **0.377203** | 0.877926 | 0.546249 |
| Final | 12,000 | 1.167999 | 2.247064 | 0.391157 | 0.870412 | 0.550524 |

The saved `checkpoint_best.pt` is selected by validation CTC and therefore
corresponds to step 9,400, not the lowest-PER report at step 11,700.

### Interpretation

- This is the strongest POSSM result recorded so far. The best pooled SBP PER
  of `0.3772` improves on the archived 12,000-step TX result (`0.467`) by about
  `0.090` absolute, or `19%` relative.
- Output-length calibration is substantially better than in the early
  collapsed runs: the best checkpoints produce about `88%` as many tokens as
  the reference, with blank-frame rate around `0.54-0.55`.
- Validation CTC was best earlier than PER. By step 12,000, training CTC was
  still falling while validation CTC had risen, indicating emerging
  overfitting even though isolated later PER improvements remained possible.
- The result does not by itself demonstrate a Brain2Text25 pretraining benefit,
  because it changes both the signal (TX to SBP) and the Stage-1 dataset pool
  relative to the archived TX reference.

The Brain2Text24-only follow-up described below was subsequently completed.
It supports a benefit from pooled pretraining, but it is not a strict isolated
comparison because Stage 1 was extended to 20,000 steps and the cache/training
path changed from FP32 to clipped FP16 plus CUDA AMP.

## 2026-08-02 Brain2Text24-only SBP FP16/AMP result

### Experiment

Stage 1 used only Brain2Text24 `competition_train`, with labels ignored, for
20,000 optimizer steps. It used 128 area-6v SBP channels from the pre-smoothed
cache after clipping SBP at `12,500` and storing it as FP16. Session-wise
normalization statistics were updated algebraically for the clipped FP16 data.
The model retained FP32 parameters and optimizer state while its expensive
forward/backward operations ran under CUDA FP16 autocast.

Stage 2 retained the same decoder architecture and 12,000-step recipe as the
pooled experiment, but used the clipped FP16 raw Brain2Text24 cache, adjusted
global statistics, online sigma-2 smoothing, and CUDA AMP.

Run artifacts:

- Stage 1:
  `sbp_only_b2t24_amp_fp16_v1/possm_stage1_sbp_b2t24_only_12k_seed7_amp_fp16_v1`
  (the run name retains `12k`, but this run was resumed through step 20,000);
- Stage 2:
  `sbp_only_b2t24_amp_fp16_v1_stage2/possm_stage2_finetune_full_sbp_only_20260801T220324Z`.

### Stage-1 reconstruction

| Measurement | Step | MSE |
|---|---:|---:|
| Validation at original stopping point | 12,000 | 0.286133 |
| Training batch at original stopping point | 12,000 | 0.236099 |
| Best sampled validation | 19,100 | **0.213823** |
| Final validation | 20,000 | 0.260469 |
| Final training batch | 20,000 | 0.222179 |

The best sampled validation MSE occurred well after step 12,000, so the extra
Stage-1 optimization was useful for reconstruction. As in the pooled run,
validation consists of newly sampled batches and individual minima are noisy.

### Stage-2 decoding

| Selection point | Step | Train CTC | Validation CTC | PER | Predicted/reference | Blank rate |
|---|---:|---:|---:|---:|---:|---:|
| Best validation CTC and PER | 11,500 | — | **2.339733** | **0.387153** | 0.881765 | 0.526961 |
| Final | 12,000 | 1.082605 | 2.612307 | 0.433390 | 0.808447 | 0.586432 |

Here the lowest validation CTC and lowest observed PER occurred at the same
report, so `checkpoint_best.pt` at step 11,500 is also the preferred PER
checkpoint. The final model is substantially worse: training CTC continued to
fall while validation CTC, PER, output-length calibration, and blank rate all
deteriorated. This is clear late Stage-2 overfitting.

### Comparison with pooled FP32 pretraining

The older pooled Brain2Text24/25 run used 12,000 Stage-1 steps with the original
FP32 caches and FP32 training. Despite receiving 20,000 Stage-1 steps, the
Brain2Text24-only FP16/AMP run reached a best PER of `0.387153`, compared with
the pooled run's best observed `0.377203`: approximately `0.010` absolute, or
2.6% relative, worse. Its final PER was `0.433390`, compared with `0.391157`
for the pooled run, and its best validation CTC was `2.339733`, compared with
`2.028183`.

This suggests that additional repetitions of Brain2Text24 did not substitute
for the diversity supplied by Brain2Text25, and that pooled Stage-1 pretraining
may also produce a model that resists late Stage-2 overfitting better. It is not
yet a clean estimate of the Brain2Text25 effect because both numerical format
and Stage-1 compute changed. The next controlled run should therefore train a
fresh pooled Brain2Text24/25 Stage 1 for 20,000 steps using the clipped FP16
caches and AMP, followed by the same 12,000-step Stage 2 recipe.

## Evaluation convention

The current reproduction follows the established
`competition_train -> competition_test` workflow and reports the test-side
metrics at every Stage-2 validation interval. Consequently, best-step metrics
are useful for matched baseline-versus-pooled comparison but are not estimates
from a test set that remained untouched during model selection. Final reporting
should state this convention explicitly.
