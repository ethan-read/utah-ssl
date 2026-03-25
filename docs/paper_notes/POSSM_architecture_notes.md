# POSSM Architecture Notes

Paper:

- `Generalizable, real-time neural decoding with hybrid state-space models`
- local source: external local paper library, file `possm.pdf` (not versioned in this repository)

## Scope Of These Notes

These notes summarize the model architecture described in the paper, with emphasis on:

- spike tokenization
- the hybrid attention + recurrent design
- how POSSM supports causal decoding
- how pretraining and finetuning are handled
- task-specific architecture variations
- architectural choices that are useful for later comparison

These are descriptive notes, not recommendations.

## High-Level Model Structure

POSSM is a hybrid neural decoding architecture that combines:

1. POYO-style individual spike tokenization
2. an input cross-attention encoder
3. a recurrent backbone
4. an output cross-attention decoder / readout

The paper presents POSSM as an attempt to keep the generalization benefits of token-based attention models while recovering:

- causal inference
- low latency
- constant-time online updates

This is the paper’s central architectural idea.

## Core Architectural Idea

The high-level computation is:

1. collect spikes in a short chunk of time
2. tokenize spikes individually
3. compress the variable-length spike set for that chunk into a fixed-size latent vector with cross-attention
4. update a recurrent hidden state using that chunk representation
5. query the most recent hidden states to predict behaviour

So POSSM splits the decoding problem into:

- local spike-set encoding with attention
- global temporal integration with a recurrent model

This is different from:

- RNNs that operate on fixed bins directly
- Transformers that attend over long histories directly

## Input Representation

### Streaming Chunks

POSSM is designed for streaming input.

The input stream is divided into short contiguous chunks. In the main experiments, the paper typically uses:

- `T_c = 50 ms`

The paper also reports:

- strong performance with `20 ms` chunks

The chunk size is therefore an important architecture and systems choice.

### Individual Spike Tokenization

Instead of binned population vectors, POSSM tokenizes individual spikes.

Each spike is represented by:

- the neural unit identity
- the spike timestamp

The token for a spike from unit `i` at time `t_spike` is written as:

- `x = (UnitEmb(i), t_spike)`

where:

- `UnitEmb(i)` is a learnable embedding for unit `i`
- spike time is encoded with RoPE

This means the model preserves precise spike timing, rather than collapsing spikes into coarse counts before the encoder.

The paper explicitly argues this helps:

- preserve temporal irregularity
- support generalization to new sessions and new unit sets

### Variable-Length Inputs

Because the number of spikes varies from chunk to chunk, each input chunk is:

- a variable-length sequence of spike tokens

This is one of the main reasons they use cross-attention as the encoder front end.

## Module 1: Input Cross-Attention Encoder

### Purpose

This module maps a variable number of spike tokens in one chunk to a fixed-size latent representation.

The paper says this is adapted from:

- POYO
- PerceiverIO-style cross-attention

### Structure

For one chunk at time index `t`, let:

- `X_t in R^(N x D)`

be the token sequence for `N` spikes in that chunk.

The model uses:

- spike tokens as keys and values
- a learnable query vector `q`

The output is a fixed-size latent `z(t)`.

The paper writes:

- `z(t) = softmax(q K_t^T / sqrt(D)) V_t`

with standard key/value projections.

So the encoder is essentially:

- single-query cross-attention from a learned latent query to a set of spike tokens

### Important Difference From POYO

The paper states a key difference from POYO:

- POYO was applied over longer contexts
- POSSM applies the encoder on short chunks, usually `50 ms`
- each chunk is mapped to a single latent vector

This is one of the most important design changes.

### Additional Transformer Block Details

Following POYO, the implementation includes:

- standard Transformer block structure
- pre-normalization layers
- feed-forward networks

So even though POSSM is “hybrid”, the input encoder is still transformer-like locally.

## Module 2: Recurrent Backbone

### Purpose

The recurrent backbone integrates local chunk representations across time.

The paper describes the hidden-state update as:

- `h(t) = f_SSM(z(t), h(t-1))`

This is the stage that enables:

- causal decoding
- online updates
- constant-time processing per new chunk

### Division Of Labor

The intended division is:

- input cross-attention captures local structure within a chunk
- the recurrent backbone accumulates longer-term context across chunks

This separation is central to POSSM’s design philosophy.

### Backbone Variants

The paper experiments with three recurrent backbones inside POSSM:

- `S4D`
- `GRU`
- `Mamba`

They note that the framework is theoretically compatible with other recurrent models as well.

### Backbone Roles

The recurrent backbone is the main mechanism for:

- maintaining state across time
- making online inference efficient
- avoiding the quadratic cost of long-context attention

## Module 3: Output Cross-Attention Readout

### Purpose

Rather than reading behaviour directly from only the latest hidden state, POSSM selects:

- the `k` most recent hidden states

and uses another cross-attention module to decode behaviour.

The paper typically uses:

- `k = 3`

### Query Design

For each time chunk, the model generates output queries that encode:

- the timestamp to predict
- a learnable session embedding

This session embedding is intended to capture latent recording-session factors.

### Why This Readout Matters

The paper emphasizes several benefits of the output cross-attention design:

- multiple behavioural outputs can be predicted per chunk
- behaviour does not need to align exactly with chunk boundaries
- the model can predict behaviour beyond the current chunk, enabling lag compensation

This is a very flexible readout design compared with a simple linear readout on the current hidden state.

## Generalization Mechanisms

The paper is strongly focused on transfer and adaptation to unseen recordings.

### Unit Identification

One finetuning mechanism is:

- `unit identification` (UI)

The key idea is:

- freeze the model weights
- initialize new unit embeddings and session embeddings for a new session
- train only those new embeddings

The paper states this usually updates:

- less than `1%` of model parameters

This is only possible because the model’s input representation is unit-embedding based rather than tied to a fixed channel ordering.

### Full Finetuning

The second finetuning mechanism is:

- gradual unfreezing followed by end-to-end finetuning

The procedure is:

1. begin with UI-style embedding-only training
2. unfreeze the full model later
3. continue full training

The paper reports that:

- full finetuning generally beats UI
- UI is still competitive and much cheaper

### Preliminary LoRA Result

The appendix reports a preliminary experiment using:

- LoRA finetuning

for `o-POSSM-GRU`

The reported result suggests LoRA can match UI while training even fewer parameters, but the paper presents this only as a preliminary direction.

## Pretraining Strategy

The larger pretrained model is called:

- `o-POSSM`

For NHP reaching tasks, the paper reports pretraining on:

- `148` sessions
- more than `670 million` spikes
- `26,032` neural units

from multiple monkey datasets and cortical areas.

An important data-handling note is that the paper intentionally minimizes preprocessing:

- no unit filtering
- no multi-unit rejection
- mixed threshold-crossing and spike-sorted units are allowed
- no standardized spike sorting across datasets
- no resampling or special processing of behavioural outputs

This reflects an architectural commitment to robustness under heterogeneous upstream data.

## Tokenization Details

The appendix notes one implementation difference from POYO:

- POSSM does not use `[START]` and `[END]` delimiter tokens

The reason given is:

- better efficiency

The paper says this did not hurt decoding performance in their experiments.

## Why POSSM Is Real-Time Friendly

The architecture is designed specifically around real-time constraints.

The paper’s real-time argument is:

- only the newest chunk must be encoded
- the recurrent state carries prior context
- inference cost does not scale with the total past sequence length

This differs sharply from POYO- or Transformer-style full-context reprocessing.

In their framing, this gives POSSM a better tradeoff between:

- accuracy
- speed
- generalization

## NHP Reaching Task Configuration

For NHP reaching, each training example consists of:

- `1 s` of spiking activity
- corresponding behavioural outputs over that same interval

The `1 s` interval is split into:

- `20` non-overlapping `50 ms` chunks

The paper trains the recurrent model causally across those chunks.

For this task:

- loss is mean squared error
- centre-out segments receive `5x` loss weighting, following POYO

### POSSM Model Sizes For NHP

Table 5 reports the following parameter counts:

- `POSSM-S4D-SS`: `0.41M`
- `o-POSSM-S4D / FT`: `4.56M`
- `POSSM-GRU-SS`: `0.47M`
- `o-POSSM-GRU / FT`: `7.96M`
- `POSSM-Mamba-SS`: `0.68M`
- `o-POSSM-Mamba / FT`: `8.96M`

This confirms that the pretrained models remain fairly small, especially compared with transformer baselines.

## Human Handwriting Configuration

The paper applies POSSM to the Willett handwriting dataset.

Important task details from the paper:

- signals are binned at `10 ms`
- POSSM achieves strong transfer from monkey motor-cortex pretraining to human handwriting decoding

The appendix says:

- they used the same POSSM architecture as in NHP reaching

So the main architecture did not need major redesign for handwriting.

The paper also describes handwriting-specific finetuning details:

- gradual unfreezing
- unit embeddings, session embeddings, and linear readout are trained first
- later unfreezing follows

This differs slightly from pure UI because the handwriting task required a task-specific readout layer.

## Human Speech Configuration

This is one of the most important sections for your purposes.

### Input Format Difference

For the speech dataset:

- only normalized spike counts were available
- exact spike times were not available in the same form

Because of that, the paper says they could not use the original POYO-style tokenization directly.

Instead:

- each multi-unit channel was treated as a neural unit
- normalized spike counts were encoded with value embeddings

So speech POSSM is not identical to spike-time POSSM.

### Chunking / Resolution

For speech:

- data were binned at `20 ms`

### Encoder Changes

For the speech task, they use:

- a GRU backbone
- an encoder with one cross-attention head
- then a self-attention module with `2` heads
- operating strictly within each `20 ms` bin

Reported dimensions:

- input dimension: `64`
- `4` latents per bin
- concatenating latents from one bin yields a `256`-dimensional vector fed into the GRU

### Output Changes

The baseline speech model used sliding windows over multiple bins.

POSSM instead:

- sets both window length and stride effectively to `1`
- uses a convolutional layer at the output to control output sequence length

This is a task-specific adaptation of the generic architecture.

### Speech Training Procedure

The paper describes a two-phase training scheme for speech:

1. pretrain the input cross-attention module, latent embeddings, and unit embeddings by reconstructing spike counts at each time bin
2. train the full POSSM model on phoneme sequences with CTC loss

This makes the speech version partially analogous to staged neural-speech pipelines like BIT, though much smaller and simpler.

### Parameter Counts For Speech

The appendix reports:

- uni-directional POSSM: `32M` parameters
- bi-directional POSSM: `86M` parameters

These are still substantially smaller than the corresponding baseline GRUs in that setup:

- uni-directional GRU baseline: `55M`
- bi-directional GRU baseline: `133M`

## Evidence For The Architecture’s Claimed Advantages

The paper presents several targeted experiments that support the architecture:

### Precise Spike Timing Matters

They explicitly ablate:

- exact spike times
- bin-level timestamps only

The paper reports that using precise spike times improves POSSM performance.

This is important because it validates the individual-spike tokenization design rather than only assuming it.

### Cross-Attention Encoder Beats Recurrent Encoder

They also replace the input cross-attention module with:

- a recurrent encoder

and report that this underperforms the standard POSSM design.

So the hybrid split appears genuinely useful:

- attention for local spike encoding
- recurrence for long-range chunk integration

### 20 ms Chunks Still Work

The appendix reports that POSSM still performs well with:

- `20 ms` chunks

This is relevant for BCI timing considerations and suggests the chunk size is flexible.

## Architectural Style

POSSM is best understood as:

- a generalizable, real-time decoder
- with tokenized neural inputs
- local attention
- global recurrence
- transfer-oriented finetuning mechanisms

It sits between two extremes:

- rigid binned RNNs
- full transformer decoders over long contexts

That hybrid position is the whole point of the paper.

## Key Architectural Tradeoffs

The paper exposes several important design tradeoffs:

- precise spike tokenization versus coarse time-binning
- local attention versus full-sequence attention
- recurrent backbone choice: S4D vs GRU vs Mamba
- UI versus full finetuning
- causal online decoding versus larger offline context
- generality across datasets versus task-specific preprocessing

These are likely the most reusable takeaways for later architecture comparison.

## Concrete Architectural Choices To Carry Forward

Without endorsing them, the paper defines a useful set of design axes:

- individual spike tokenization versus binned population vectors
- learnable unit embeddings
- RoPE time encoding for spikes
- local cross-attention encoder versus recurrent encoder
- recurrent backbone type: GRU, S4D, Mamba
- single latent per chunk versus multiple chunk outputs
- output cross-attention over recent hidden states
- session embeddings in the readout
- UI, full finetuning, or LoRA-style adaptation
- exact spike timing versus bin-level timing
- recurrent online decoding versus full-context transformer decoding

These are especially relevant because they are orthogonal to many of the design choices in BIT and Cortical-SSM.

## Short Summary

POSSM is a hybrid neural decoding architecture that:

- tokenizes individual spikes using unit embeddings plus RoPE time encoding
- compresses each short chunk of spikes with POYO-style input cross-attention
- integrates information across chunks with a recurrent backbone such as GRU, S4D, or Mamba
- predicts behaviour by querying the most recent hidden states with output cross-attention
- supports efficient transfer through unit-embedding/session-embedding adaptation

Its main architectural contribution is the combination of:

- flexible spike-token input processing
- recurrent causal state updates
- lightweight transfer mechanisms

to produce a decoder that aims to be both generalizable and real-time capable.
