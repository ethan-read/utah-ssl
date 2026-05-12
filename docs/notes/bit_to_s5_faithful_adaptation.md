# Faithful BIT-to-S5 Adaptation Guide

This document gives detailed instructions for implementing a version of `BIT` that is as faithful as possible to the paper while replacing the transformer neural encoder with an `S5` backbone.

The guiding rule is:

- change the sequence backbone
- keep the rest of the pipeline as close to BIT as possible unless the backbone swap forces a change

This is not the same as designing the best possible `S5` speech model. It is a controlled adaptation recipe for later comparison.

## Goal

The target model should preserve the main BIT recipe:

1. self-supervised masked reconstruction on large cross-task / cross-species Utah-array data
2. phoneme-level supervised fine-tuning with `CTC`
3. sentence-level fine-tuning with an `LLM` decoder
4. contrastive neural-text alignment during the sentence stage

The main architectural substitution is:

- replace the bidirectional transformer encoder with an `S5` sequence encoder

Everything else should be preserved unless there is a clear incompatibility.

## Fidelity Rules

If the goal is a faithful adaptation, keep the following BIT choices fixed:

- same neural inputs: `TX + SBP` for downstream speech decoding when available
- same pretraining asymmetry: `TX` only during SSL if some pretraining datasets lack `SBP`
- same `20 ms` base binning
- same temporal patching concept, with `patch_size = 5` as the starting default
- same three training stages
- same phoneme intermediate supervision with `CTC`
- same sentence-level `LLM` interface through an `MLP` projector
- same sentence-level loss structure: `L_CE + L_contrastive`
- same lightweight subject-specific boundary adaptation idea

Avoid introducing the following in the first faithful version:

- new SSL objectives
- extra convolutional front ends
- multiscale branches
- heavy subject/session adapters beyond boundary layers
- direct sentence decoding without phoneme fine-tuning
- causal-only constraints if the comparison target is BIT as written

## High-Level Architecture

The faithful `S5` version should be:

1. input features in `20 ms` bins
2. temporal patchification into `100 ms` patch tokens
3. patch embedding into model width
4. `S5` sequence backbone over patch tokens
5. task-specific output head depending on stage

The stage-specific heads are:

- SSL stage: reconstruction head back to neural patch space
- phoneme stage: linear phoneme classifier with `CTC`
- sentence stage: `MLP` projector into `LLM` embedding space, plus modality aligner

## Step 1: Preserve BIT Input Handling

Use the same neural feature policy as BIT:

- downstream speech stages:
  - threshold crossings / spike counts: `TX`
  - spike band power: `SBP`
- SSL pretraining:
  - use `TX` only if `SBP` is unavailable in part of the pretraining corpus

Use the same temporal discretization:

- bin width: `20 ms`

Apply the same general normalization policy:

- z-score across days / sessions to reduce Utah-array nonstationarity

Do not change the feature definition during the first adaptation. The point is to isolate the backbone swap.

## Step 2: Keep Temporal Patching

For a faithful adaptation, preserve BIT-style time patching before the backbone.

Start with:

- `patch_size = 5`
- patch duration: `5 x 20 ms = 100 ms`

The patched input should be:

- raw sequence: `(T, C)`
- patched sequence: `(T / 5, 5C)`

Why keep patching in the `S5` version:

- BIT uses it
- it reduces sequence length by about `5x`
- it changes the sequence unit from `20 ms` bins to `100 ms` neural chunks
- it makes the comparison cleaner because the backbone is the main changed component

For the first faithful version, use non-overlapping patches. This matches the simplest reading of BIT.

## Step 3: Replace The Transformer With An S5 Backbone

This is the core substitution.

The BIT transformer does:

- sequence modeling over patch embeddings
- bidirectional context aggregation
- stage-shared hidden representation learning

The `S5` backbone should fill the same role:

- accept a sequence of patch embeddings
- output a sequence of latent states with one state per patch
- feed those latent states into the stage-specific head

### Output Interface

Require the `S5` encoder to return:

- a full hidden-state sequence, not just a pooled vector

This is necessary because:

- SSL reconstruction needs patch-level outputs
- phoneme `CTC` needs sequence-level outputs
- the sentence stage needs a sequence of neural tokens for the projector and `LLM`

### Width And Depth

Do not immediately search for the best `S5` size.

For the first faithful adaptation:

- choose one `S5` width and depth that keeps the encoder in the same rough scale as BIT
- target rough parity in representational budget rather than exact parameter matching

The BIT transformer core is reported at about `7M` parameters. Use that as a rough budget reference.

## Step 4: Preserve Subject-Specific Boundary Layers

BIT has direct paper evidence for subject-specific boundary layers in SSL pretraining.

For the faithful `S5` version, implement:

- subject-specific read-in / patch-embedding modules
- subject-specific reconstruction read-out modules in SSL

The shared `S5` core should remain common across subjects.

### Recommended Interpretation

Use the following decomposition:

- subject-specific read-in:
  - maps flattened input patches to model-width embeddings
- shared `S5` backbone:
  - learns the common sequence dynamics
- task-specific head:
  - reconstruction, phoneme classification, or sentence projection

For SSL reconstruction, use subject-specific read-out layers that map hidden states back to the neural patch feature space.

### What To Keep Across Stages

For a faithful implementation, the safest staged policy is:

- keep the subject-specific input/read-in side across SSL and phoneme fine-tuning
- replace the SSL reconstruction read-out when moving to phoneme fine-tuning
- optionally keep subject-specific input adaptation into the sentence stage too

This preserves the clearest BIT-like idea:

- shared encoder core
- lightweight subject-specific boundary adaptation

## Step 5: Reproduce BIT’s SSL Stage With S5

This stage should remain conceptually unchanged except for the backbone.

### Objective

Pretrain on large cross-task / cross-species Utah-array data using:

- masked temporal-patch reconstruction
- `MSE` loss

### Masking

Keep the BIT-style masking scheme:

- patch-level masking, not raw-bin masking
- replace selected patches with a learnable mask token or mask embedding
- allow contiguous masked spans
- hold overall masking ratio fixed

### Reconstruction Head

The SSL head should:

- take the `S5` hidden state for each patch
- map it back to the original patch feature dimension
- reconstruct the original continuous neural patch values

This head predicts continuous neural data, not labels.

### Important Implementation Detail

The backbone input unit is a patch token, but the reconstruction target remains continuous neural data.

That means:

- patch tokenization is the encoder interface
- reconstruction is the SSL target

Those are compatible and should not be conflated.

## Step 6: Decide How To Handle Bidirectionality

This is the most important design choice forced by the backbone swap.

BIT’s transformer encoder is explicitly:

- offline
- bidirectional

A plain `S5` implementation is often used causally. If you switch to strictly causal `S5`, the comparison stops being a faithful BIT adaptation and becomes a new model family.

### Faithful Default

If the goal is fidelity to BIT, use an offline bidirectional `S5` encoder.

Good ways to do that include:

- a forward `S5` plus backward `S5` with concatenated or summed states
- a bidirectional `S5` block wrapper if your implementation supports it

The first faithful comparison should preserve:

- access to past and future context within a sentence

### Alternative

You may later run a second experiment with causal `S5`, but that should be labeled clearly as:

- BIT-inspired causal adaptation

not:

- faithful BIT adaptation

## Step 7: Replace The SSL Head With A Phoneme CTC Head

After SSL pretraining:

- remove the masking module
- remove the reconstruction head
- keep the pretrained encoder weights
- attach a phoneme classification head

The phoneme head should be:

- a linear layer over each patch-level hidden state

The output vocabulary should match BIT:

- phonemes
- silence token
- `CTC` blank token

Train this stage with:

- `CTC` loss

### Why The Head Must Change

The SSL reconstruction head outputs:

- continuous-valued neural patch predictions

The phoneme head outputs:

- discrete phoneme logits

These are different tasks with different target spaces, so the reconstruction head should not be reused as-is.

### What Carries Over

The key transfer hypothesis is:

- SSL gives the `S5` encoder a useful latent representation of neural structure
- phoneme supervision then reshapes that representation toward speech-relevant information

## Step 8: Preserve The Cascaded Decoder Baseline

If the goal is a faithful BIT adaptation, do not skip the cascaded evaluation.

After phoneme fine-tuning:

- decode phoneme logits with the same `5`-gram LM setup used for the speech benchmark
- include optional `OPT` rescoring if you are reproducing the full BIT cascaded pipeline

This matters because BIT uses the cascaded path to answer:

- is the encoder itself strong before adding the `LLM` decoder?

That question remains useful for the `S5` adaptation.

## Step 9: Preserve The Sentence-Level LLM Stage

After phoneme fine-tuning:

- keep the phoneme-shaped `S5` encoder
- attach the same `MLP` projector into `LLM` embedding space
- attach the same modality aligner
- fine-tune the encoder, projector, and `LoRA` adapters

### Projector

Use the same default as BIT:

- `Linear -> ReLU -> Linear`

Do not replace this with a heavier interface in the first faithful version.

### LLM Decoder

Use the same sentence-generation setup as BIT:

- neural hidden-state sequence projected into `LLM` token embedding space
- prompt inserted after neural tokens
- autoregressive next-token training

At inference:

- provide only neural embeddings plus prompt
- generate text autoregressively

### Prompting

Preserve the same prompt templates BIT used when possible:

- neural modality:
  - `<neural_activity>#</neural_activity>`
  - `decode the above neural activity into an English sentence:`
- audio modality:
  - `<|audio_bos|>#<|audio_eos|>`
  - `transcribe the above audio into an English sentence:`

Start with the neural-modality variant, since BIT reports it performs slightly better.

## Step 10: Preserve The Contrastive Alignment Objective

Do not drop the contrastive term in the first faithful `S5` version.

The sentence-stage objective should be:

- `L_total = L_CE + L_contrastive`

where:

- `L_CE` is autoregressive next-token cross-entropy
- `L_contrastive` is sentence-level neural-text alignment

### Modality Aligner

Replicate BIT’s aligner structure:

- mean-pool neural token embeddings over time
- mean-pool text embeddings over tokens
- apply modality-specific linear projections into a shared latent space
- `L2` normalize both projected vectors
- optimize a symmetric `InfoNCE`-style loss

This should be added on top of the sentence decoder, not used as a replacement for it.

## Step 11: Keep The Same Fine-Tuning Granularity

The sentence-stage update set should stay as BIT-like as possible.

Update:

- `S5` encoder
- `MLP` projector
- modality aligner
- `LoRA` adapters in the `LLM`

Do not fully fine-tune the whole `LLM` in the first faithful version.

## Step 12: Recommended Experiment Order

Run the adaptation in the following order:

1. implement `S5` encoder with BIT-style patch input and sequence output
2. implement SSL masked reconstruction with subject-specific read-in/read-out
3. verify reconstruction works on a small subset
4. swap in the phoneme `CTC` head and fine-tune
5. evaluate cascaded phoneme-to-text decoding
6. attach `MLP` projector + modality aligner + `LLM`
7. fine-tune sentence generation with `L_CE + L_contrastive`
8. compare against BIT-style transformer baseline under matched input and training setup

Do not start by attaching the `LLM` directly to a random or SSL-only `S5` encoder. BIT strongly suggests the phoneme stage matters.

## Step 13: What Would Count As A Fair Comparison

To claim you have faithfully adapted BIT to `S5`, the following should remain matched as closely as possible:

- same datasets and train / val splits
- same `20 ms` base binning
- same patch size
- same `TX` / `SBP` feature policy
- same staged training order
- same phoneme `CTC` supervision
- same cascaded decoding setup
- same sentence-level projector type
- same prompt family
- same contrastive alignment loss
- same `LLM` backbone family and `LoRA` strategy

Then the main changed factor is:

- transformer sequence backbone versus `S5` sequence backbone

## Step 14: What Not To Conclude Too Early

If the first `S5` version underperforms BIT, do not immediately conclude that `S5` is worse for the task.

Check first whether the gap is caused by:

- causal `S5` being compared against bidirectional BIT
- different patching
- dropping subject-specific boundary layers
- skipping the phoneme stage
- changing the projector or prompt design
- removing the contrastive loss
- mismatched pretraining corpus or feature set

Those changes would break the faithfulness of the comparison.

## Step 15: Minimum Spec For A Faithful S5-BIT

The minimum acceptable implementation should have:

- `20 ms` bins
- `patch_size = 5`
- subject-specific read-in
- shared offline `S5` backbone
- subject-specific SSL reconstruction read-out
- phoneme linear head with `CTC`
- cascaded decoding baseline
- sentence-level `MLP` projector
- modality aligner with contrastive loss
- `LLM` fine-tuned with `LoRA`

If any of these are missing, label the result as:

- partial BIT-inspired `S5` adaptation

rather than:

- faithful BIT-to-`S5` adaptation

## Practical Summary

The cleanest faithful port of BIT into an `S5` setting is:

- keep BIT’s inputs
- keep BIT’s patching
- keep BIT’s staged training
- keep BIT’s subject-specific boundary adaptation
- keep BIT’s phoneme intermediate task
- keep BIT’s `LLM` interface
- keep BIT’s contrastive sentence loss
- swap only the transformer sequence model for an offline sequence-output `S5`

That gives the fairest answer to the question:

- what changes if BIT’s transformer encoder is replaced by `S5` while the rest of the system stays the same?
