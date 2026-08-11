# Future Prediction SSL Results

This document summarizes the main Brain2Text24 future-prediction SSL runs for
the `tx_sbp` / `Mamba` experiment.

## Setup

- dataset: `brain2text24`
- features: `tx_sbp` (`128 TX + 128 SBP = 256` dims)
- backbone: causal `Mamba`
- tokenization: `raw_bin`
- effective bin size: `40 ms`
  - implemented as `segment_bins=24` at native `20 ms` with
    `temporal_bin_stride=2`
- effective context: `12` bins = `480 ms`
- forecast target: next `1` effective bin
- normalization: per-session per-channel z-scoring
- default objective: `Huber` on all channels + optional variance-match penalty
- current best line: slimmed `Mamba` model after removing the dead HF embedding
  table

## Model Size Correction

The original HF `MambaModel` construction was allocating an unused token
embedding table because we feed `inputs_embeds` directly.

- old total params: `15,051,520`
- unused embedding params: `12,871,680`
- corrected total params: `2,180,096`
- corrected encoder params: `2,114,304`
- forecast head params: `65,792`

This brought the model into the same parameter regime as the paper's
`~1.95M`-parameter forecaster.

## Main Runs

### 1. Baseline 40 ms / 3-horizon / no variance regularizer

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_diag5k`
- steps: `5k`
- config:
  - `future_bins=3`
  - `variance_match_weight=0.0`
- final metrics:
  - `loss = 0.1909`
  - `h1_mae = 0.4776`
  - `h3_mae = 0.4825`
  - `pred_std = 0.2539`
  - `target_std = 0.7176`
  - `pred_to_target_std_ratio = 0.3539`
  - `zero_baseline_mae = 0.5324`
- interpretation:
  - better than zero baseline
  - clearly under-dispersed
  - not collapsed to literal zero

### 2. 40 ms / 3-horizon / variance-match `0.05`

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_varmatch_diag5k`
- steps: `5k`
- config:
  - `future_bins=3`
  - `variance_match_weight=0.05`
- final metrics:
  - `base_loss = 0.1951`
  - `loss = 0.2085`
  - `h1_mae = 0.4823`
  - `h3_mae = 0.4861`
  - `pred_std = 0.3471`
  - `pred_to_target_std_ratio = 0.4837`
- interpretation:
  - improved variance match
  - hurt forecast accuracy too much

### 3. 40 ms / 3-horizon / variance-match `0.01`

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_varmatch001_diag5k`
- steps: `5k`
- config:
  - `future_bins=3`
  - `variance_match_weight=0.01`
- final metrics:
  - `base_loss = 0.1912`
  - `loss = 0.1950`
  - `h1_mae = 0.4774`
  - `h3_mae = 0.4825`
  - `pred_std = 0.2759`
  - `target_std = 0.7176`
  - `pred_to_target_std_ratio = 0.3844`
- interpretation:
  - better tradeoff than `0.05`
  - still multi-horizon and still somewhat smoothed

### 4. 40 ms / 1-horizon / variance-match `0.01` with large accidental model

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_diag5k`
- steps: `5k`
- config:
  - `future_bins=1`
  - `variance_match_weight=0.01`
  - old bloated HF `Mamba` construction
- final metrics:
  - `base_loss = 0.1895`
  - `loss = 0.1930`
  - `h1_mae = 0.4765`
  - `pred_std = 0.2917`
  - `target_std = 0.7171`
  - `pred_to_target_std_ratio = 0.4068`
  - `zero_baseline_mae = 0.5329`
- interpretation:
  - best result among the original `5k` runs
  - switching from 3-horizon to 1-horizon helped

### 5. Mixed loss experiment: `TX Poisson + SBP Huber`

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_txpoisson_sbp_huber_var001_diag5k`
- steps: `5k`
- config:
  - `future_bins=1`
  - `variance_match_weight=0.01`
  - `tx_loss_type=poisson_nll`
  - `sbp_loss_type=huber`
- final metrics:
  - `base_loss = 0.3744`
  - `loss = 0.3750`
  - `h1_mae = 0.5141`
  - `pred_std = 0.8898`
  - `target_std = 0.7171`
  - `pred_to_target_std_ratio = 1.2408`
  - `zero_baseline_mae = 0.5329`
- interpretation:
  - did beat the zero baseline
  - worse than the Huber-only 1-horizon run
  - over-shot the target variance
  - not the preferred objective so far

### 6. Slimmed model / `10k` / Huber-only

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim10k`
- steps: `10k`
- config:
  - slimmed `~2.18M` model
  - `future_bins=1`
  - `variance_match_weight=0.01`
  - `tx_loss_type=huber`
  - `sbp_loss_type=huber`
- final metrics:
  - `base_loss = 0.1798`
  - `loss = 0.1836`
  - `h1_mae = 0.4705`
  - `pred_std = 0.2655`
  - `target_std = 0.6939`
  - `pred_to_target_std_ratio = 0.3826`
  - `zero_baseline_mae = 0.5219`
- interpretation:
  - better than the older bloated-model `5k` run
  - justified continuing training

### 7. Slimmed model / `12k` / Huber-only

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim12k`
- resumed from `slim10k`
- final metrics:
  - `base_loss = 0.1766`
  - `loss = 0.1799`
  - `h1_mae = 0.4599`
  - `pred_std = 0.3020`
  - `target_std = 0.7076`
  - `pred_to_target_std_ratio = 0.4268`
  - `zero_baseline_mae = 0.5342`
- interpretation:
  - better than `10k`
  - best final-step result observed so far

### 8. Slimmed model / `20k` / Huber-only

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim20k`
- resumed from `slim12k`
- final-step metrics:
  - `base_loss = 0.1833`
  - `loss = 0.1865`
  - `h1_mae = 0.4659`
  - `pred_std = 0.3148`
  - `target_std = 0.7278`
  - `pred_to_target_std_ratio = 0.4326`
  - `zero_baseline_mae = 0.5419`
- interpretation:
  - still better than `10k`
  - slightly worse than the `12k` final-step metric
  - training after `12k` did not monotonically improve the final checkpoint

## Best Checkpoint From The `20k` Run

The best checkpoint inside the `slim20k` run was earlier than the final step.

- checkpoint: `checkpoint_best.pt`
- checkpoint step: `16700`
- checkpoint kind: `best`

Local forecast-correlation evaluation over `256` validation batches
(`45,056` valid forecast timesteps):

- overall flattened prediction-vs-target `r`: `0.3848`
- population-mean time-series `r`: `0.7829`
- per-channel mean `r`: `0.3613`
- per-channel median `r`: `0.3373`
- per-channel std of `r`: `0.1256`
- per-channel min / max `r`: `0.1327 / 0.7690`

By feature group:

- `TX` mean `r`: `0.3083`
- `TX` median `r`: `0.2913`
- `SBP` mean `r`: `0.4143`
- `SBP` median `r`: `0.3987`

First `16` channel `r` values:

`[0.3706, 0.5148, 0.3284, 0.4310, 0.4596, 0.2849, 0.3861, 0.4531, 0.3681, 0.3167, 0.4922, 0.3345, 0.5551, 0.4672, 0.4720, 0.3091]`

## Main Conclusions So Far

1. Future prediction on `tx_sbp` is not collapsing to zero.
   - The model consistently beats the zero baseline.
   - Channelwise and population-level correlations are clearly above zero.

2. Single-horizon forecasting works better than the original 3-horizon setup.
   - Moving from `future_bins=3` to `future_bins=1` improved `h1_mae`.

3. Mild variance regularization helps more than aggressive regularization.
   - `variance_match_weight=0.01` is preferable to `0.05` in these runs.

4. The mixed `TX Poisson + SBP Huber` objective is not currently better.
   - It increased dispersion but degraded forecast accuracy.

5. The model-size correction was important.
   - The original `15M` model was mostly an unused HF embedding table.
   - The corrected `~2.18M` model is closer to the paper and performs better.

6. The current best model should be taken from the `slim20k` run's
   `checkpoint_best.pt`, not its final checkpoint.
   - best checkpoint step: `16700`
   - best final-step run among kept summaries: `slim12k`
   - best retained checkpoint artifact: `slim20k/checkpoint_best.pt`

## Current Recommended Artifact

Use:

- run: `future_pred_mamba_b2t24_txsbp_40ms_ctx12bins_h1_var001_slim20k`
- checkpoint: `checkpoint_best.pt`
- checkpoint step: `16700`

This is the best preserved artifact for downstream phoneme probing and further
analysis.
