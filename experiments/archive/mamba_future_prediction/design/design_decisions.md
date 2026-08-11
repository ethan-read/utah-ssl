# Future Prediction SSL Design Decisions

This document tracks the design decisions for adapting the
`mambaforecastsst`-style future-prediction idea to Brain2Text24 Utah-array
phoneme decoding.

The goal is to move slowly, keep the contract explicit, and avoid mixing too
many experiment changes at once.

## Working Goal

Train a causal `Mamba` encoder on self-supervised future prediction over
Brain2Text24 `tx_sbp` neural features, then test whether the resulting latent
states support a stronger lightweight downstream phoneme decoder than a matched
non-pretrained baseline.

The closest analogue to the paper is not "beat Willett." It is:

1. pretrain a causal forecaster
2. freeze or mostly freeze it
3. train a lightweight readout on top of its latents
4. show the readout is better than a matched baseline without that pretraining

## Decisions Already Made

### Backbone

- status: decided
- choice: `Mamba`
- notes:
  - use the existing generic `ssm_ssl` Mamba path as the implementation anchor
  - keep the model causal

### Feature mode

- status: decided
- choice: `tx_sbp`
- notes:
  - follow the repo's current area-6v policy
  - expected input dim is `256` total: `128` `TX` + `128` `SBP`

### Platform

- status: leaning decided
- choice: `Modal`
- notes:
  - reuse the existing persistent-volume pattern from the current Modal stage-1
    workflow
  - likely GPU request should follow the current documented preference:
    `L40S` first, `RTX-PRO-6000` fallback, unless a stronger reason appears

### Forecast horizon

- status: decided at high level
- choice: predict the next `3` bins
- notes:
  - Brain2Text24 bins are `20 ms`, so this means predicting the next `60 ms`
  - use the native `20 ms` cache binning
  - motivation: phoneme-relevant structure is likely finer-grained than the
    `50 ms` bins used in the reference paper
  - the exact target format is still open: direct multi-horizon prediction vs
    recursive next-bin prediction

## Open Decisions

These are the main items we still need to settle before implementation.

### 1. What exactly is the SSL target?

- status: mostly decided
- why it matters:
  - the paper predicts next-bin spike counts with Poisson NLL
  - our inputs are continuous `tx_sbp` features, not count data

Options:

- predict the next `3` raw feature bins directly
- predict the next `3` normalized feature bins
- predict future temporal patches rather than raw bins
- predict a latent target from a stop-gradient target encoder instead of raw
  features

Decision:

- direct prediction of the next `3` normalized feature bins

Reason:

- this is the closest analog to the paper while still matching our continuous
  feature regime
- it avoids needing a second target encoder in the first pass
- it avoids recursive rollout instability

### 2. What loss should we use?

- status: decided
- why it matters:
  - Poisson NLL is not appropriate for `tx_sbp` features

Options:

- mean squared error on future bins
- smooth L1 / Huber
- Gaussian negative log-likelihood with learned variance
- cosine or correlation-style loss on future vectors
- weighted combination, for example `MSE + cosine`

Decision:

- use `Huber`

Reason:

- still a plain point-prediction objective
- more robust than `MSE` to occasional large transients or outlier bins
- appropriate for continuous normalized `tx_sbp` targets

Implementation note:

- the default stage-1 path remains `Huber` on all channels
- the code now also supports an experimental mixed objective:
  `PoissonNLL(softplus(tx_logits), tx_raw)` for `tx` plus `Huber` on normalized
  `sbp`
- that mixed path reconstructs raw nonnegative `tx` targets from the normalized
  cache batches using the stored per-session z-score stats
- this keeps the input pipeline unchanged while making the `tx` term more
  paper-like

### 3. What representation do we decode from?

- status: open
- why it matters:
  - in the paper, the readout uses a lightweight decoder over the forecaster's
    output representation
  - for phoneme `CTC`, the closest analogue is likely a linear decoder over
    hidden states, not a full nonlinear downstream stack

Options:

- use the encoder hidden state before the prediction head
- use the prediction head outputs directly
- concatenate hidden state with predicted future features

Current lean:

- decode from the encoder hidden states and discard the future-prediction head
  for the first downstream probe

Reason:

- this is the cleanest analogue to the paper's claim that forecasting-trained
  latents become more decodable
- it separates "latent quality" from "can a larger downstream model recover the
  signal anyway"

### 4. What input tokenization should stage 1 use?

- status: decided
- why it matters:
  - this repo already shows that temporal interface choices strongly affect
    downstream phoneme decoding

Options:

- `raw_bin`
- `temporal_patch`
- `causal_conv_stem`

Decision:

- use `raw_bin`

Reason:

- this is the closest analogue to the paper
- it preserves the native `20 ms` time resolution
- it avoids adding a learned temporal front-end before we know whether future
  prediction on raw bins helps

Follow-up choice still needed:

- if we use `temporal_patch`, do we predict future bins from each patch state
  or future patches aligned to the token rate?

### 5. What patch size and stride should stage 1 use?

- status: decided by tokenization choice

Options:

- match the supervised Willett-style decoder exactly: `patch_size=14`,
  `patch_stride=4`
- use denser stage-1 patching such as `5/5` or `5/1`
- use one setup for SSL and another for stage 2

Decision:

- no patching in the first probe because stage 1 uses `raw_bin`

Main concern:

- if stride is too coarse, a linear `CTC` readout may fail for alignment reasons
  even when the latent states are useful

### 6. What pretraining dataset scope should we use?

- status: open

Options:

- Brain2Text24 only
- multi-dataset Utah pretraining that includes Brain2Text24
- multi-dataset pretraining excluding Brain2Text25 as in current BIT-style runs

Current lean:

- start with Brain2Text24 only for the first clean experiment

Reason:

- reduces confounding
- keeps the first question narrow: does future prediction help on the exact
  target dataset?

Counterargument:

- if the method needs more data to work well, a Brain2Text24-only first pass may
  understate its value

### 7. What normalization and smoothing regime should stage 1 use?

- status: decided at first-pass level
- why it matters:
  - this repo distinguishes carefully between raw cache, pre-smoothed cache,
    session stats, and split/global stats

Options:

- raw cache + no smoothing
- raw cache + online smoothing
- pre-smoothed cache + matching session stats

Decision:

- stage 1 should stay as close as possible to the paper's simplicity:
  - use the native raw Brain2Text24 cache
  - do not use the pre-smoothed cache
  - do not add extra stage-1 runtime smoothing
  - forecast the native 20 ms `tx_sbp` bins directly

Remaining nuance:

- our features are continuous and training will likely still need a stable
  normalization convention
- so "do what they did" maps to "avoid extra smoothing and feature engineering,"
  not to literally skipping all normalization questions

Current first-pass interpretation:

- raw cache, no extra smoothing, minimal normalization needed for stable Mamba
  training

Normalization decision:

- use per-channel z-scoring for stage 1

Reason:

- this is the minimal normalization needed to keep `tx` and `sbp` scales from
  dominating the regression loss unevenly
- it stays much closer to the paper's simplicity than adding smoothing or more
  elaborate feature engineering

### 8. What segment length should stage 1 see?

- status: open

Why it matters:

- predicting `3` future bins is local, but the representation may need much
  longer context to help speech decoding

Options:

- reuse the generic SSL default segment length
- choose a longer causal context specifically for speech

Current lean:

- keep enough context for speech-related dynamics, not just next-step
  autocorrelation

### 9. How should the prediction head be structured?

- status: open

Options:

- linear head from hidden state to `3 * input_dim`
- small MLP head to `3 * input_dim`
- one head per horizon

Current lean:

- start with a simple linear projection

Reason:

- isolates whether the encoder learns anything useful
- avoids credit assignment getting hidden inside a large forecast head

### 10. How should the downstream readout consume the pretrained encoder?

- status: decided for the first probe

Options:

- freeze the encoder and train only a linear phoneme head
- freeze most of the encoder and tune only a thin adapter plus linear head
- initialize the encoder and fine-tune end to end
- run both a frozen linear probe and a full fine-tune

Decision:

- first result should use a completely frozen encoder with a linear `CTC` probe
  on the latent sequence
- full fine-tuning is secondary

Reason:

- this is the closest analogue to the paper's lightweight decoder result
- it separates "representation is useful" from "optimization is helped"

### 11. What is the main success metric?

- status: open, but partially constrained by repo norms

Candidates:

- validation `CTC` loss
- phoneme error rate
- both, with one designated as checkpoint-selection metric

Current lean:

- downstream `PER` is the real decision metric
- but track `CTC` closely because existing runs show they do not always align

### 12. What baselines do we require before claiming success?

- status: open but critical

Minimum sensible baselines:

- frozen linear `CTC` head on pretrained Mamba latents
- the same frozen linear `CTC` head on non-pretrained latents
- the same linear `CTC` head directly on raw features if feasible at matched
  frame rate
- optionally masked-reconstruction pretraining under the same probe protocol

Important note:

- the paper's matched-context raw baseline logic does not transfer directly to
  our `CTC` setting
- our equivalent rigor is to keep the readout lightweight and matched while
  varying only whether the Mamba latents were made useful by future prediction

### 13. What is the primary downstream claim?

- status: open

Candidates:

- paper-style claim: future-prediction-pretrained Mamba latents support a
  stronger lightweight phoneme decoder
- stronger secondary claim: the same pretraining also improves end-to-end
  phoneme fine-tuning

Current lean:

- prioritize the paper-style lightweight-decoder claim first
- treat end-to-end fine-tuning as follow-up evidence, not the first target

### 14. What should the first run be optimized for?

- status: open

Options:

- smallest possible proof of plumbing
- strongest plausible research run

Current lean:

- do a tiny plumbing run first
- then one clean, defensible main run

Reason:

- Modal setup plus a new objective plus `Mamba` plus `tx_sbp` is too many
  moving pieces to debug in one expensive launch

### 15. What Modal job contract do we want?

- status: open

Need to decide:

- run script path and naming
- cache root inside volume
- output root inside volume
- whether we need new stats artifacts uploaded
- default GPU type
- timeout and retry behavior
- whether stage 2 fine-tuning is in the same Modal entrypoint or a separate one

Current lean:

- follow the existing two-volume Modal pattern
- keep stage 1 and stage 2 as separate jobs

## Current Recommended First-Pass Contract

This is not final, but it is the most conservative starting point right now.

- backbone: `Mamba`
- feature mode: `tx_sbp`
- direction: `causal`
- platform: `Modal`
- tokenization: `raw_bin`
- bin size: `20 ms`
- target: next `3` future bins (`60 ms` total horizon)
- target format: direct multi-horizon prediction
- loss: `Huber`
- cache variant: raw cache
- stage-1 smoothing: none beyond native cache contents
- stage-1 normalization: per-channel z-scoring
- transfer object: encoder latent states, not forecast outputs
- first downstream evaluation: completely frozen encoder plus linear phoneme
  `CTC` probe
- first comparison: same lightweight readout with vs without future-prediction
  pretraining
- first deployment style: separate stage-1 and stage-2 jobs

## Questions To Answer Next

To make implementation concrete, the next decisions should probably be:

1. What is the smallest fair baseline set for the first result?
2. What exact linear `CTC` readout shape should we use?
