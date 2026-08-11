# Timestep-Flexible SSM Tests And Results

This note is the local experiment log for the active
`experiments/archive/timestep_flexible_ssm` branch.

It tracks:

- completed baseline evaluations
- the exact comparison definitions used
- the next planned training tests

## Active Artifacts

- notebook: `experiments/archive/timestep_flexible_ssm/notebooks/timestep_flexible_s5.ipynb`
- package: `experiments/archive/timestep_flexible_ssm`
- current partial S5 run: `timestep_flexible_s5_tx_only_colab`
- Willett GRU baseline run: `willett_tx_only_area6v_colab`

## Completed Baseline Snapshot

Context:

- dataset: `brain2text24`
- features: area-6v `tx_only`
- labels: canonical phoneme `CTC` targets
- canonical cache bin size: `20 ms`
- shared real-time patch geometry: `280 ms` patch, `80 ms` stride

### Current Comparison Table

| model | bin size | val_ctc_bpphone | val_phoneme_error_rate |
| --- | --- | ---: | ---: |
| timestep-flexible S5 | `20 ms` | `2.9076` | `0.5545` |
| timestep-flexible S5 | `40 ms` | `2.9150` | `0.5565` |
| Willett GRU | `20 ms` | `1.8476` | `0.3724` |
| Willett GRU | `40 ms` | `1.8561` | `0.3735` |

### Source Notes

S5 source:

- artifact root: `/content/drive/MyDrive/utah_ssl/outputs/timestep_flexible_ssm/timestep_flexible_s5_tx_only_colab`
- summary source: `summary.json`
- values above are the best-selection metrics from the saved run summary

GRU source:

- artifact root: `/content/drive/MyDrive/utah_ssl/outputs/willett_reconstruction/willett_tx_only_area6v_colab`
- sources used: `progress.jsonl` and `checkpoint_best.pt`
- the notebook `20 ms` re-eval matches the saved best `20 ms` GRU metrics:
  - `val_ctc_bpphone = 1.8476455153996616`
  - `val_phoneme_error_rate = 0.3724146472360979`

### Caveat

The GRU number comes from a mature supervised baseline run. The current
timestep-flexible S5 result is still an early exploration run, so this is not
yet an apples-to-apples training maturity comparison.

## Completed GRU `40 ms` Evaluation Definition

The saved Willett GRU checkpoint was trained on canonical `20 ms` inputs with:

- `patch_size = 14` bins
- `patch_stride = 4` bins

That corresponds to:

- `280 ms` patch duration
- `80 ms` patch hop

For the `40 ms` evaluation, the notebook uses this compatibility path:

1. Rebin validation inputs from canonical `20 ms` to `40 ms`.
2. Keep patch duration and hop fixed in milliseconds.
3. Use:
   - `20 ms`: `patch_size = 14`, `patch_stride = 4`
   - `40 ms`: `patch_size = 7`, `patch_stride = 2`
4. Resample each `40 ms` patch token back to the checkpoint's expected
   `14`-bin width before passing it into the saved GRU.
5. Keep evaluation-time preprocessing aligned with the Willett recipe:
   - no training noise
   - same effective smoothing in milliseconds
   - train-derived normalization stats in the matching bin space

This avoids the confound where raw `14/4` bins at `40 ms` would silently
change the real-time frontend to `560 ms` patches with `160 ms` stride.

## Interpretation So Far

The GRU retains almost all of its validation performance under the matched-ms
`40 ms` evaluation:

- `CTC`: `1.8476 -> 1.8561`
- `PER`: `0.3724 -> 0.3735`

That suggests the current `40 ms` perturbation is mild when:

- real-time patch geometry is preserved
- patch tokens are resampled back to the trained input width
- smoothing and normalization remain matched in milliseconds

This should not yet be interpreted as true timestep invariance. It is a
carefully constructed compatibility evaluation.

## Next Planned Training Tests

### Test 1: Mixed-Bin Training

Goal:

- train on a split dataset containing both `20 ms` and `40 ms` bins

S5 behavior:

- the model is explicitly timestep-aware
- each sequence keeps its own effective timestep metadata

GRU baseline behavior:

- the model remains structurally unchanged
- each `40 ms` bin is expanded into two identical `20 ms` bins before the GRU
  frontend so the checkpoint/train-time token geometry stays compatible

Main question:

- does mixed-bin exposure improve cross-bin robustness for the timestep-aware
  S5 relative to the patched GRU baseline?

### Test 2: Missing-Bin Training

Goal:

- train and evaluate under dropped-timestep conditions

S5 behavior:

- the model receives the actual timestep information directly
- missing bins are represented by the larger effective elapsed time between
  available observations

GRU baseline behavior during training:

- missing bins are filled by interpolation before they are fed to the GRU

GRU baseline behavior during inference:

- when a new observation is missing, the previous bin or bins are repeated
  until a new real observation arrives

Main question:

- does explicit timestep conditioning help the S5 degrade more gracefully than
  an imputation-based GRU baseline when observations arrive irregularly?

### Test 3: Future-Bin Prediction After `20 ms` Training

Goal:

- train both models on canonical `20 ms` data
- then evaluate how well they can predict bins that lie `20-100 ms` into the
  future

Shared setup:

- both models are trained on standard `20 ms` supervised data first
- future-prediction evaluation is added afterward as a separate test, not as a
  replacement training objective for the current baseline
- horizons of interest are:
  - `20 ms`
  - `40 ms`
  - `60 ms`
  - `80 ms`
  - `100 ms`

S5 behavior:

- use the timestep-aware state model after standard `20 ms` training
- test whether its learned dynamics support better short-horizon prediction at
  variable future offsets

GRU baseline behavior:

- keep the baseline trained on canonical `20 ms`
- evaluate future-bin prediction with a GRU-compatible rollout or prediction
  head defined on the same `20 ms` training setup

Main question:

- after identical `20 ms` training, does the timestep-aware S5 retain more
  useful predictive state for short future horizons than the GRU baseline?

## Immediate Implementation Notes

For the next round, keep the comparisons auditable:

- preserve the Willett split policy and normalization conventions unless a
  test explicitly targets those choices
- keep patch duration and hop defined in milliseconds
- record exactly how `40 ms` duplication, interpolation, and repeated-bin
  inference are implemented for the GRU baseline
- track train mixture proportions and missing-bin schedules in this note once
  they are fixed
- define future-bin targets precisely for the prediction test:
  - whether the target is raw firing rate, normalized firing rate, or patched
    token content
  - whether prediction is teacher-forced, autoregressive, or based on a simple
    readout from the trained hidden state
