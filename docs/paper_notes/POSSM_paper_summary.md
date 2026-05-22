# POSSM Paper Summary

Paper:

- `Generalizable, real-time neural decoding with hybrid state-space models`
- local source: `/Users/home/Documents/School/thesis/papers/possm.pdf`

Related note:

- architecture-focused companion: [POSSM_architecture_notes.md](/Users/home/thesis/utah-ssl/docs/paper_notes/POSSM_architecture_notes.md)

## Scope Of These Notes

These notes summarize the paper with emphasis on the speech experiment.

Non-speech results are only included briefly for context.

## Short Takeaway

POSSM is a hybrid neural decoder that tries to get the best of both worlds:

- Transformer-like flexibility for handling variable spike inputs and cross-session pretraining
- recurrent / state-space efficiency for causal, low-latency decoding

The paper's central claim is that this hybrid design keeps decoding quality near strong attention-based baselines while making online inference much cheaper and easier to scale to real-time BCI settings.

## What Problem The Paper Is Solving

The authors argue that a useful neural decoder should satisfy three things at once:

1. strong decoding accuracy
2. causal, low-latency inference
3. good transfer to new sessions, subjects, and tasks

Their framing is:

- classic RNN-style decoders are efficient but tied to rigid binned inputs and often generalize poorly
- large attention models generalize better, especially with pretraining, but are expensive for long-context online decoding
- a hybrid model may preserve the transfer benefits of tokenized attention while recovering constant-time recurrent updates

## Main Idea

POSSM combines three pieces:

1. POYO-style spike tokenization
2. a cross-attention encoder that compresses each short chunk of spikes
3. a recurrent backbone that carries state forward over time

In the main setup:

- neural activity is processed in `50 ms` chunks
- each chunk is turned into a fixed-size latent representation
- the recurrent backbone updates its hidden state once per chunk
- the decoder reads out behavior from the most recent hidden states

The architectural note in [POSSM_architecture_notes.md](/Users/home/thesis/utah-ssl/docs/paper_notes/POSSM_architecture_notes.md) covers the module details in more depth.

## Why The Input Representation Matters

Instead of starting from fixed population bins, POSSM tokenizes individual spikes using:

- unit identity
- spike timestamp

This matters because it gives the model:

- variable-length inputs per chunk
- precise spike timing
- a cleaner path to transfer across sessions with different unit sets

The authors lean heavily on this tokenization scheme for both pretraining and adaptation.

## Recurrent Backbone Choices

The POSSM framework is paired with several recurrent backbones:

- `GRU`
- `S4D`
- `Mamba`

Conceptually, the division of labor is:

- cross-attention handles local structure inside each chunk
- the recurrent backbone handles long-range temporal integration

In the reaching experiments, the best pretrained transfer results are often from `o-POSSM-S4D`, while `GRU` remains important in the speech setting.

## Pretraining And Adaptation Strategy

The paper is not only about architecture. A major contribution is the adaptation recipe.

The authors describe two transfer modes:

1. `Unit identification (UI)`: freeze the main model and learn only new unit embeddings plus session embeddings for unseen units / sessions
2. `Full finetuning (FT)`: start from UI, then gradually unfreeze the rest of the model

UI is the cheaper option:

- typically updates less than `1%` of parameters

FT is the stronger option:

- usually gives better transfer when moving to a new animal, dataset, or task

This is one of the main practical lessons of the paper: the tokenization scheme is what makes this kind of lightweight adaptation possible.

## Datasets And Evaluation

The paper evaluates POSSM on three task families:

1. non-human-primate reaching
2. human imagined handwriting
3. human attempted speech

### 1. NHP Reaching

Used mainly to show pretraining and transfer across sessions / animals.

Main point:

- pretrained `o-POSSM` improves transfer, especially with full finetuning

### 2. Human Handwriting

Used mainly as a cross-species transfer result.

Main point:

- NHP pretraining also helps on a human handwriting task

### 3. Human Speech

Task:

- decode variable-length phoneme sequences from attempted speech

Dataset:

- `24` sessions from one participant with speech deficits
- four `64`-channel microelectrode arrays
- premotor cortex and Broca's area
- multi-unit activity binned at `20 ms`
- trial lengths range from `2` to `18 s`

Why speech is a special case in the paper:

- sequences are much longer than the `1 s` contexts used by POYO-style models
- this makes full attention-based decoding much less attractive computationally
- the causal recurrent backbone is especially useful here

## How They Changed POSSM For Speech

The speech experiment is not a pure drop-in use of the spike-time POSSM setup.

Because the public speech dataset only exposed normalized spike counts rather than spike times:

- they could not use the original POYO-style spike-time tokenization directly
- they treated each multi-unit channel as a neural unit
- they embedded normalized spike-count values instead of individual spike events

They also changed the output stage:

- instead of the standard output cross-attention readout, they used a `1D` strided convolution
- this controlled the length and emission frequency of the output phoneme sequence

For speech they used POSSM with:

- a `GRU` backbone
- a cross-attention encoder with one head
- a local self-attention block with two heads inside each `20 ms` bin
- `4` latents per bin
- `64`-dimensional input embeddings

At the encoder output:

- the `4` latents from a bin were concatenated into a `256`-dimensional vector
- that vector was then fed into the GRU

## Working Encoder Choice For Binned Rates

The paper is ambiguous about the exact token construction for binned speech inputs, so for reproduction we should treat the following as a principled implementation choice rather than a verbatim statement from the paper.

Recommended tokenization for bin `t` and unit / channel `i`:

- one token per `(t, i)`
- `x_{t,i}` = normalized spike count or normalized firing rate for unit `i` in bin `t`
- `e_i` = learned unit embedding of size `d`
- `v_{t,i} = ValueEncoder(x_{t,i})`, where `ValueEncoder` is a small continuous map from a scalar to `R^d`
- `h_{t,i} = LayerNorm(e_i + v_{t,i})`

Practical reproduction details:

- use all channels in every bin, including zero-activity channels
- keep time at the bin level; do not concatenate a separate time embedding into the token
- for the simplest baseline, let `ValueEncoder` be a single linear layer from `1 -> d`
- if needed, upgrade `ValueEncoder` to a small `MLP`
- all tokens in the same `20 ms` bin share the same bin time; temporal integration is mainly handled across bins by the `GRU`

Why this is the safest starting point:

- it preserves unit identity explicitly
- it treats firing rate as a continuous value rather than a discrete symbol
- it is closer to the POSSM speech setup than event-style POYO tokenization

Other reasonable choices to test briefly:

- `concat + projection`: `h_{t,i} = LayerNorm(W[e_i || v_{t,i}])`
- FiLM / gating: let the value modulate the unit embedding multiplicatively
- avoid lookup embeddings over quantized firing-rate values as the default, since that throws away the continuity of the counts

## Working Attention Block Choice

The speech paper describes the encoder at a high level, but not every block detail. For reproduction we will use the following default.

Per `20 ms` bin:

- exactly one cross-attention block
- exactly one self-attention block
- standard residual connections plus `FFN` after each attention block, following POYO
- a small dropout, e.g. `0.1`
- no `RoPE` in the per-bin block for our working implementation, since all channels in a bin are treated as simultaneous measurements

Recommended dimensions:

- model width `d = 64`
- latents per bin `L = 4`
- cross-attention: `1` head with head dimension `64`
- self-attention: `2` heads with head dimension `32` each

Recommended tensor flow:

- binned input: `X in R^(B x T x U)`
- tokenized input: `H in R^(B x T x U x 64)`
- reshape bins for parallel per-bin processing: `H_bin in R^((B*T) x U x 64)`
- learned latent queries: `Z0 in R^(4 x 64)`, repeated across bins to `Z in R^((B*T) x 4 x 64)`
- cross-attention from `4` latents to `U` unit tokens gives `R^((B*T) x 4 x 64)`
- apply residual + `FFN`
- self-attention over the `4` latent tokens gives `R^((B*T) x 4 x 64)`
- apply residual + `FFN`
- concatenate the `4` latents to get `R^((B*T) x 256)`
- reshape back to `R^(B x T x 256)` and feed that to the `GRU`

Implementation note:

- all channels in a bin are treated as simultaneous measurements, so the per-bin attention is mainly compressing cross-channel structure
- temporal modeling is mainly handled across bins by the `GRU`
- whether channel masking is needed for this dataset should be checked from the saved cache rather than assumed
- for the active cached `brain2text24` data, the feature axis should be structurally dense after the area-6v migration (`128` `TX` + `128` `SBP`), so we do not currently need a missing-unit mask for structural absence

## Working Stage-1 Reconstruction Choice

The paper says:

- in phase 1, they trained the `input cross-attention module` together with the latent and unit embeddings
- the objective was reconstructing spike counts at each individual time bin

The most conservative interpretation is:

- the `1D` strided convolution head belongs to stage 2 phoneme decoding, not stage 1 reconstruction
- stage 1 is a local per-bin reconstruction task, not a sequence-decoding task
- stage 1 should not require `CTC`, output-sequence length control, or the phoneme readout stack

Because the paper only explicitly names the input cross-attention module in phase 1, the paper text is underspecified. Our working choice is:

- use the per-bin tokenization described above
- use the `4` learned latent queries
- run one cross-attention block followed by one self-attention block
- keep the POYO-style residual + `FFN` structure around both attention blocks
- do not rely on the recurrent decoder or the strided-conv phoneme head for this phase

For a practical reconstruction head, use the simplest local decoder:

- concatenate the `4` latents to get `R^((B*T) x 256)`
- predict the same bin's channel vector with a linear layer or small `MLP`
- for `brain2text24`, reconstruct both `TX` and `SBP`, so the target shape is `R^((B*T) x 512)`

Default stage-1 loss:

- use `MSE`
- run both normalized-input and raw-input variants as separate experiments if we want to match the paper's broader robustness comparisons
- for now, keep the architecture fixed and vary only the input / target scaling between those runs

Recommended paper-faithful starting point:

- no masking in stage 1
- reconstruct the full same-bin `TX+SBP` vector
- treat masked reconstruction as a later experimental extension rather than the default POSSM reproduction

## Working Conv Head Choice For Stage 2 Decoding

For stage 2 phoneme decoding, the paper says the standard output cross-attention module was replaced by a `1D` strided convolution to control output-sequence length.

Recommended working interpretation:

- the per-bin encoder still produces `4 x 64` latent vectors
- concatenate them to get one `256`-dimensional vector per bin
- run the `GRU` across bins
- apply a temporal `Conv1d` over the sequence of `GRU` states
- map the conv outputs to phoneme logits and train with `CTC`

Recommended tensor flow:

- encoder output to recurrent model: `E in R^(B x T x 256)`
- `GRU` output: `G in R^(B x T x H_g)`
- transpose for `Conv1d`: `R^(B x H_g x T)`
- temporal strided convolution: `R^(B x H_c x T')`
- transpose back: `R^(B x T' x H_c)`
- linear phoneme readout: `R^(B x T' x V)`
- apply `log_softmax` over `V` and train with `CTC`

Interpretation of the conv:

- kernel size controls how much local temporal context each output step sees
- stride controls how often the model emits output steps, and therefore mostly controls the output length
- example: with `20 ms` bins and stride `4`, the model emits one output roughly every `80 ms`

Reasonable default if exact details are unknown:

- use the same recurrent scale as the Willett/Card baseline they reference:
  - `5` GRU layers
  - hidden width `768`
  - recurrent dropout `0.4`
- use a causal unidirectional `GRU`
- use a causal `Conv1d` over time
- exact conv kernel / stride / padding should be treated as an empirical choice to test
- a linear layer after the conv to produce phoneme logits

## First Implementation Spec

This is the first concrete end-to-end spec to implement for a Utah-array adaptation of the POSSM speech setup.

Data and targets:

- dataset: `brain2text24`
- bin size: `20 ms`
- input modalities: `TX + SBP`
- feature width after the area-6v cache migration: `256 = 128 TX + 128 SBP`
- no structural missing-unit mask
- phoneme vocabulary: use the cached `CTC` targets from the canonical dataset

Runs:

- normalized-input run
- raw-input run
- keep the architecture identical across both runs

Per-bin tokenization:

- one token per channel per bin
- learned unit embedding of width `64`
- value encoder: start with a single linear map from scalar to `64`
- token construction: `h_{t,i} = LayerNorm(e_i + ValueEncoder(x_{t,i}))`
- no `RoPE`

Per-bin encoder:

- reshape from `R^(B x T x 512 x 64)` to `R^((B*T) x 512 x 64)`
- use `4` learned latent queries of size `64`
- one cross-attention block:
  - `1` head
  - head dimension `64`
  - residual + `FFN`
  - dropout `0.1`
- one self-attention block:
  - `2` heads
  - head dimension `32` each
  - residual + `FFN`
  - dropout `0.1`
- concatenate the `4` latents to get one `256`-dimensional vector per bin

Stage 1 reconstruction pretraining:

- input to reconstruction head: `R^((B*T) x 256)`
- reconstruction head: start with a single linear layer `256 -> 512`
- target: reconstruct the full same-bin `TX+SBP` vector
- loss: `MSE`
- no masking in the first POSSM-faithful implementation

Stage 2 phoneme decoding:

- initialize from the stage-1 encoder checkpoint
- restore time axis to `R^(B x T x 256)`
- recurrent backbone:
  - unidirectional `GRU`
  - `5` layers
  - hidden width `768`
  - recurrent dropout `0.4`
- conv readout starting point:
  - causal `Conv1d`
  - input channels `768`
  - output channels `768`
  - kernel size `14`
  - stride `4`
  - left padding chosen to preserve causal behavior
- phoneme head:
  - transpose back to time-major hidden states
  - linear layer from `768 -> V`
  - `log_softmax`
  - `CTC` loss

Interpretation of the stage-2 output rate:

- with `20 ms` input bins and stride `4`, phoneme logits are emitted about every `80 ms`
- kernel size `14` gives each output step a local temporal receptive field of about `280 ms`

Items we still expect to test empirically:

- conv kernel / stride / padding
- linear vs small `MLP` reconstruction head
- linear vs small `MLP` value encoder
- normalized vs raw input scaling

## Speech Training Recipe

The authors report that the best speech results came from a two-phase procedure:

1. pretrain the input cross-attention module plus latent / unit embeddings to reconstruct spike counts at each time bin
2. then train the full model on phoneme targets using `CTC` loss

This is one of the most important details for us because it is the closest thing in the paper to a self-supervised or representation-learning stage before sequence decoding.

## Main Results

### Decoding Quality

The broad pattern is:

- POSSM performs well across all three settings
- pretraining helps
- the speech setting is where the hybrid long-context story matters most

Speech results reported as phoneme error rate (`PER`, lower is better):

- `GRU (no aug.)`: `39.16`
- `POSSM-GRU (no aug.)`: `29.70`
- `GRU`: `30.06`
- `S4D`: `35.99`
- `Mamba`: `32.19`
- `POSSM-GRU`: `27.32`
- `GRU (mult.)`: `21.74`
- `POSSM-GRU (mult.)`: `19.80`

So in speech, POSSM-GRU clearly beats the recurrent baselines they compare against.

### Efficiency

The paper repeatedly argues that POSSM is useful because it is fast enough for real-time use.

Reported CPU inference times:

- single-session POSSM: about `2.44 ms/chunk`
- pretrained `o-POSSM`: about `5.65 ms/chunk`

The authors note this is comfortably below a rough real-time BCI target of `<= 10 ms` decoding latency.

For the speech model specifically, the architecture also reduces parameter count relative to the GRU baseline:

- uni-directional GRU baseline: `55M` params
- uni-directional POSSM: `32M`
- bi-directional GRU baseline: `133M`
- bi-directional POSSM: `86M`

## What Seems Most Important For Our Later Adaptation Discussion

If we want to adapt this paper's speech experiment to our data later, the most relevant takeaways are:

1. POSSM's main advantage is not just the recurrent block, but the combination of local token compression plus recurrent state updates.
2. Their speech setup already departs from the original spike-time formulation because the dataset only had binned count features.
3. They used an explicit two-phase training recipe, with count reconstruction before CTC-based phoneme decoding.
4. The recurrent formulation makes sense when trials are long and causal decoding matters.
5. The paper is optimistic about multimodal extensions and more principled self-supervised pretraining, which lines up well with our interests.

## Limits And Caveats

Some useful caution points:

- their speech experiment does not use the full spike-time tokenization story because the public dataset did not provide spike times
- much of the transfer story is strongest for reaching and handwriting, not speech
- the paper demonstrates a reconstruction-style pretraining step for speech, but not a full standalone self-supervised learning framework
- the online-decoding argument is motivated carefully, but the evaluations in the paper are still offline

## Bottom Line

The paper is best read as a strong systems-and-modeling paper about how to combine:

- flexible neural tokenization
- pretraining-friendly adaptation
- recurrent causal decoding

For our purposes, the most important part is probably the speech section's hybrid recipe:

- compact local encoder
- recurrent long-context model
- reconstruction pretraining
- CTC decoding

That is the part we should compare most directly against our Utah speech setup.
