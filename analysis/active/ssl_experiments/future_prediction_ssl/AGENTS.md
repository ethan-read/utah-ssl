# Objective

This folder is for active future-prediction self-supervised learning
experiments on Brain2Text24 that adapt the
`mambaforecastsst`-style forecasting idea to Utah-array phoneme decoding.

# Current Research Direction

The first goal in this folder is not to beat the strongest full supervised
decoder recipe. It is to test the paper-style claim that causal forecasting can
produce more decodable latent states.

The current intended comparison path is:

- dataset: `brain2text24`
- features: area-6v `tx_sbp`
- training resolution: native `20 ms` bins
- SSL target: direct prediction of the next `3` bins (`60 ms` total horizon)
- model family: causal `Mamba`
- stage-1 input mode: `raw_bin`
- stage-1 loss: `Huber`
- stage-1 cache: raw cache
- stage-1 smoothing: none beyond native cache contents
- stage-1 normalization: per-channel z-scoring
- first downstream evaluation: frozen encoder plus lightweight linear phoneme
  `CTC` probe
- first baseline: the same lightweight probe on a non-pretrained `Mamba`

# Working Assumptions

Use the reference paper as the structural anchor for the first experiment:

- causal forecaster
- native bin-rate input, not patched input
- no extra smoothing pipeline
- simple point-prediction objective
- lightweight downstream readout over frozen latents

Treat the first result as a latent-decodability test, not as proof that future
prediction is the best full speech-decoding recipe.

Keep the stage-1 contract simple and auditable:

- raw Brain2Text24 cache
- native `20 ms` `tx_sbp` bins
- direct multi-horizon prediction
- no pre-smoothed cache
- no silent switch to a reconstruction or contrastive objective

# Fixed First-Pass Decisions

These choices should not be changed casually inside this folder without
updating the design notes and documenting why.

- backbone: `Mamba`
- direction: `causal`
- feature mode: `tx_sbp`
- tokenization: `raw_bin`
- forecast horizon: next `3` bins
- target format: direct multi-horizon prediction, not recursive rollout
- loss: `Huber`
- cache root family: raw cache
- stage-1 smoothing: none
- stage-1 normalization: per-channel z-scoring
- first downstream probe: completely frozen encoder
- first downstream head: lightweight linear phoneme `CTC` readout
- preferred platform for substantial runs: `Modal`

The design discussion for these choices lives in:

- `analysis/active/ssl_experiments/future_prediction_ssl/design_decisions.md`

# Open Design Questions

The first implementation should keep these questions visible rather than burying
them in notebook state:

- whether the first downstream probe should decode from encoder hidden states or
  from forecast-head outputs
- the exact shape of the linear phoneme `CTC` head
- the minimum fair baseline set beyond the non-pretrained frozen-encoder probe
- whether Brain2Text24-only pretraining is sufficient before trying
  multi-dataset Utah pretraining
- whether full end-to-end fine-tuning should help after the frozen-probe result
  is established

Default to the simpler, paper-analogue comparison first:

- frozen latent probe before full fine-tuning
- one clean forecasting objective before ablations
- one clean baseline before wider sweep logic

# Reference Baselines

The primary conceptual baseline is the paper's structure:

- forecasting-trained latent representation
- lightweight downstream decoder
- matched decoder without forecasting pretraining

Within this repo, useful reference implementations are:

- `analysis/active/ssl_experiments/ssm_ssl`
- `analysis/active/ssl_experiments/ssl_core`
- `analysis/active/ssl_experiments/timestep_flexible_ssm`
- `analysis/active/ssl_experiments/willett_reconstruction`

These are references for reusable components and evaluation utilities, not
instructions to inherit Willett patching or reconstruction-stage assumptions
into this folder's first experiment.

# Modal Direction

Substantial future-prediction runs in this folder should prefer the established
Modal pattern already used elsewhere in the repo:

- persistent cache volume
- persistent outputs volume
- separate stage-1 and downstream-probe jobs
- GPU preference aligned with current documented Modal practice unless updated

Keep Modal workflow notes explicit when new scripts are added so future runs do
not have to rediscover cache layout, stats expectations, or archive upload
steps.

# Cleanup Direction

Prefer a small, scriptable experiment package over notebook-only state.

Keep:

- explicit stage-1 forecasting code
- explicit frozen-probe evaluation code
- tests that protect horizon alignment, target construction, and frozen-encoder
  behavior
- notes that record why a change was made to the stage-1 contract

Avoid:

- one-off notebooks that silently change tokenization, horizon definition, or
  normalization
- run names that do not encode horizon, feature mode, and frozen-vs-finetuned
  probe style
- mixing multiple new ideas into the first result without a documented reason
