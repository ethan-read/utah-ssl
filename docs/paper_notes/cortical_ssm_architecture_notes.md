# Cortical-SSM Architecture Notes

Paper:

- `Cortical-SSM: A Deep State Space Model for EEG and ECoG Motor Imagery Decoding`
- local source: external local paper library, file `cortical_ssm.pdf` (not versioned in this repository)

## Scope Of These Notes

These notes summarize the model architecture in the paper, with emphasis on:

- signal representation
- main architectural modules
- why the authors chose S5
- model fusion and classifier design
- training objective and reported hyperparameters
- architectural choices that may matter later for comparison

These are descriptive notes, not recommendations.

## High-Level Model Structure

Cortical-SSM is a classification model for EEG and ECoG motor-imagery decoding.

Its core design is:

1. a frequency-feature front end called `Wavelet-Convolution`
2. two parallel SSM towers:
   - `Frequency-SSM`
   - `Channel-SSM`
3. a fusion stage plus classifier head

The paper’s main claim is that the model captures dependencies jointly across:

- temporal domain
- spatial / electrode domain
- frequency domain

Unlike many transformer-based EEG/ECoG models, the authors explicitly avoid temporal patchification or other temporal compression before long-range modeling.

## Input Representation

The model input is:

- `X in R^(M x T)`

where:

- `M` is the number of electrodes
- `T` is the sequence length

For preprocessing, the paper states:

- EEG and ECoG signals are downsampled to `250 Hz`
- signals are used directly as model inputs
- no artifact-removal or noise-removal procedure across electrodes is applied in preprocessing

So the model is designed to work from fairly raw multichannel time series.

## Core Architectural Idea

The architecture separates two complementary modeling views after frequency feature extraction:

- frequency-wise modeling of spatio-temporal patterns
- channel-wise modeling of temporal-frequency patterns

This is the main structural novelty of the paper.

Instead of only modeling time, or only modeling channels, the model creates two parallel branches that organize the same intermediate representation in two different ways.

## Module 1: Wavelet-Convolution

### Purpose

This module extracts frequency-domain features while trying to preserve both:

- interpretability
- learnability

The authors frame this as a compromise between:

- deterministic frequency transforms, which are interpretable but rigid
- learned convolutional frequency features, which are flexible but less interpretable

### Structure

The Wavelet-Convolution module has two parallel branches:

- `E-Branch` for deterministic frequency extraction
- `A-Branch` for adaptive learned frequency extraction

For each electrode signal `x_m in R^T`, the output is:

- `x_tilde_m in R^(F x T)`

and the full output is:

- `X_tilde in R^(M x F x T)`

The fusion equation in the paper is:

- `0.5 * LayerNorm(CWT(x_m)) + 0.5 * LayerNorm(Conv1D(x_m))`

So the two branches are simply averaged after temporal-dimension normalization.

### E-Branch

The deterministic branch uses:

- continuous wavelet transform (`CWT`)

The mother wavelet is:

- Morlet wavelet

The filter bank spans:

- `F` frequency components
- frequency range `(f_min, f_max)`

### A-Branch

The adaptive branch uses:

- a `1D` convolutional layer

The kernel length is set as:

- `K = f_sample / 2`

following EEGNet.

Since the sampling rate is `250 Hz`, this implies:

- kernel length `125`

if the implementation uses the same effective sampling rate stated in preprocessing.

### Normalization Choice

An important design choice is that normalization is applied:

- along the temporal dimension

not across variables/channels.

The authors explicitly justify this by arguing that cross-variable normalization can inject noise in multivariate time-series settings when shared events appear at different times across variables.

That is a notable architectural decision because it reflects a strong inductive bias about asynchronous neural responses across electrodes.

### Reported Hyperparameters

The implementation details state:

- frequency dimension `F = 50`
- frequency range `(f_min, f_max) = (1 Hz, 100 Hz)`

### Architectural Role

This module is not just a fixed preprocessing step. It is part deterministic transform and part learned front end.

The paper’s ablation claims that using both branches together performs better than using:

- CWT alone
- Conv1D alone
- STFT-based variants

So the intended lesson is that hybrid frequency extraction mattered materially for performance.

## Module 2: Frequency-SSM

### Purpose

The Frequency-SSM branch models:

- spatio-temporal interactions within each frequency component

The authors motivate this from motor-imagery neuroscience:

- task-related power changes are frequency-specific
- those changes localize to functionally relevant cortical regions

So this branch treats each frequency slice as its own multivariate sequence over electrodes and time.

### Structure

The module contains:

- `L` stacked blocks

Each block includes:

- layer normalization
- a Deep SSM
- a feed-forward network

The branch processes each frequency component independently.

The paper writes the block update as:

- `u_f^(l+1) = FFN(SSM(u_tilde_f^(l)) + u_tilde_f^(l))`

where:

- `u_tilde_f^(l)` is the temporally normalized representation for frequency component `f`

This is effectively an SSM block with residual connection, followed by feed-forward processing.

### SSM Choice

The paper explicitly chooses:

- `S5`

and rejects several alternatives conceptually.

The reasoning is:

- they want a time-invariant SSM rather than time-varying Mamba-style selection
- they want a MIMO model rather than SISO
- EEG/ECoG are multivariate continuous signals, so preserving inter-variable dependencies is important

This is one of the most important architectural choices in the paper.

The authors claim S5 is a better fit than Mamba-style architectures for this setting because:

- time-varying selection may be harmful for continuous signals
- MIMO structure is a natural fit for multi-electrode data

### What This Branch Learns

Conceptually, this branch learns:

- how patterns evolve over time within each frequency band
- while preserving cross-electrode relationships inside that frequency band

## Module 3: Channel-SSM

### Purpose

The Channel-SSM branch models:

- temporal-frequency dependencies within each electrode

This is complementary to Frequency-SSM.

If Frequency-SSM asks:

- “within one frequency band, how do spatial and temporal patterns evolve?”

then Channel-SSM asks:

- “within one electrode, how do frequency and time patterns evolve together?”

### Structure

This module also contains:

- `L` stacked blocks

Each block includes:

- layer normalization
- a Deep SSM
- a feed-forward network

The block update is written as:

- `v_m^(l+1) = FFN(SSM(LayerNorm(v_m^(l))) + LayerNorm(v_m^(l)))`

where:

- `v_m^(l)` is the representation for electrode `m`

So the branch mirrors Frequency-SSM structurally, but reorganizes the representation by electrode rather than by frequency.

### What This Branch Learns

Conceptually, this branch learns:

- localized temporal-frequency dynamics per electrode

The paper argues that this helps capture electrode-specific intensity changes associated with motor imagery.

## Fusion And Classifier Head

After the two branches, the outputs from:

- `U^(l)` from Frequency-SSM
- `V^(l)` from Channel-SSM

are fused.

The fusion is:

- average-pool `U`
- average-pool `V`
- concatenate the two pooled outputs
- pass the result through an FFN classifier

The model outputs:

- predicted class probabilities over `N` action classes

The paper says:

- average pooling is performed along the temporal dimension
- this follows the approach used in S4

### Loss

The classification loss is:

- cross-entropy

So the model is straightforwardly supervised at the top level.

## Why S5 Instead Of Attention Or Mamba

This paper is unusually explicit about architectural choice among sequence models.

### Attention

The authors argue that transformer-style attention often requires:

- temporal patching
- temporal compression

to remain computationally feasible on long EEG/ECoG sequences.

They view this as potentially harmful because it can lose fine-grained temporal dependencies.

### Mamba / Time-Varying SSMs

They note that many recent EEG models use:

- Mamba

but argue that:

- input-dependent time-varying selection may be less suitable for continuous signals
- prior work suggests this type of mechanism can be disadvantageous on some continuous signal tasks

### S5

They choose S5 because it provides:

- time-invariant dynamics
- MIMO state-space structure
- explicit suitability for multivariate continuous signals

This choice is also backed by their ablation table, where S5 outperforms:

- attention
- S4-LegS
- Mega
- Mamba
- Mamba-2

## Reported Ablation Conclusions

The paper’s architecture ablations support three main design decisions:

### Wavelet-Convolution Ablation

They compare:

- STFT only
- CWT only
- Conv1D only
- STFT + Conv1D
- CWT + Conv1D

The best-performing configuration is:

- `CWT + Conv1D`

So their preferred front end is explicitly hybrid.

### Temporal-Dependency Architecture Ablation

They compare the SSM backbone choice inside Frequency-SSM and Channel-SSM using:

- Attention
- S4-LegS
- Mega
- Mamba
- Mamba-2
- S5

The best-performing option is:

- `S5`

### Module-Wise Ablation

They ablate:

- Wavelet-Convolution
- Frequency-SSM
- Channel-SSM

The full model with all three modules performs best.

Among the three, the paper says:

- Wavelet-Convolution contributes the largest single performance gain

## Training Details

The implementation section reports:

- optimizer: `AdamW`
- beta values: `beta1 = 0.9`, `beta2 = 0.999`
- learning rate: `1e-4`
- batch size: `8`
- epochs: `100`

Reported architecture settings:

- frequency dimension `F = 50`
- frequency range `(1 Hz, 100 Hz)`
- stacked block number `L = 2` for both Frequency-SSM and Channel-SSM

Model scale:

- approximately `0.93 million` trainable parameters
- approximately `2.34 billion` multiply-add operations

Training hardware reported:

- Nvidia GeForce RTX 4090 with `24 GB`

This is a relatively small model by current sequence-model standards.

## Architectural Style

Cortical-SSM is not a foundation-model or pretraining-heavy architecture.

Its style is:

- task-specific
- relatively compact
- strongly structured by domain priors
- built for multivariate neural signal classification rather than text generation

The model’s design is much more modular and inductive-bias-heavy than BIT.

## What Makes This Architecture Distinct

The most distinctive choices are:

- no temporal patchification before sequence modeling
- explicit frequency-domain front end
- hybrid deterministic + learned frequency extraction
- two parallel sequence-modeling branches organized by frequency and by channel
- use of S5 rather than transformer attention or Mamba
- direct interpretability claims tied to model structure

This is not a generic sequence backbone with minimal preprocessing. It is a carefully structured architecture built around specific assumptions about EEG/ECoG signals.

## Architectural Limitations Mentioned By The Paper

The paper notes that although the model captures integrated dependencies across time, space, and frequency, it still has limitations, including:

- limited cross-domain integration
- no explicit mechanism for subject- or session-level domain shift

That matters for transfer-learning questions, because the model is not designed around cross-session robustness in the way a foundation-model pipeline might be.

## Concrete Architectural Choices To Carry Forward

Without endorsing them, the paper defines a useful set of design axes:

- raw-input model versus explicit frequency-domain front end
- deterministic frequency transform versus learned frequency extraction versus hybrid
- STFT versus CWT
- one-branch versus multi-branch modeling
- organize latent dynamics by frequency versus by channel
- transformer attention versus Deep SSM
- time-varying SSM versus time-invariant SSM
- SISO versus MIMO state-space formulation
- separate branch modeling plus fusion versus unified single-tower modeling
- temporal pooling strategy
- compact supervised model versus larger pretrained model

These are useful comparison dimensions for later architectural synthesis.

## Short Summary

Cortical-SSM is a compact supervised classification architecture for EEG/ECoG motor-imagery decoding built around:

- a hybrid `Wavelet-Convolution` front end combining `CWT` and `1D` convolution
- a `Frequency-SSM` branch that models spatio-temporal patterns independently per frequency band
- a `Channel-SSM` branch that models temporal-frequency patterns independently per electrode
- `S5` state-space blocks in both branches
- temporal average pooling, branch concatenation, and an FFN classifier trained with cross-entropy

Its core architectural claim is that explicit parallel modeling across frequency-wise and electrode-wise views, combined with a time-invariant MIMO SSM backbone, is better suited to EEG/ECoG motor-imagery decoding than both transformer-based patchified models and Mamba-style SSM baselines.
