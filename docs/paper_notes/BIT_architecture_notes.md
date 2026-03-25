# BIT Architecture Notes

Paper:

- `A Cross-Species Neural Foundation Model for End-to-End Speech Decoding`
- local source: external local paper library, file `BIT.pdf` (not versioned in this repository)
- metadata title from PDF: `A cross-species neural foundation model for end-to-end speech decoding`

## Scope Of These Notes

These notes summarize the model architecture described in the paper, with emphasis on:

- neural input representation
- encoder design
- decoder design
- training stages
- key hyperparameters
- architectural choices that may matter for later comparison

They are descriptive notes, not a recommendation.

## High-Level Model Structure

BIT is organized as a three-stage neural-to-text system:

1. a transformer-based neural encoder
2. an intermediate projector / modality-alignment stack
3. an LLM decoder that generates text autoregressively

The paper presents BIT as both:

- a cascaded decoder, where the encoder is trained for phoneme decoding and then paired with a 5-gram LM plus OPT rescoring
- an end-to-end decoder, where encoder representations are passed directly into an audio-LLM or text-LLM for sentence generation

The paper’s main architectural claim is that a pretrained transformer encoder plus an audio-LLM decoder works better than prior end-to-end RNN-based approaches.

## Neural Input Representation

For the speech datasets used in fine-tuning:

- neural features include thresholded spike counts and spike-band power
- both are binned at `20 ms`
- features are z-scored across days to reduce Utah-array nonstationarity

For pretraining:

- the encoder is pretrained on approximately `367` hours of Utah-array recordings
- this includes approximately `98` hours of human data and approximately `269` hours of monkey data
- the human pretraining corpus includes the Brain-to-Text speech datasets later used in supervised decoding, but with labels discarded during SSL
- because SBP is not available in all pretraining datasets, pretraining uses thresholded spikes only

This creates an asymmetry:

- pretraining uses threshold crossings only
- downstream speech fine-tuning uses threshold crossings plus SBP

One important experimental detail is therefore:

- BIT does expose the encoder to Brain-to-Text '24/'25 neural recordings during SSL pretraining
- this is not "fine-tune only on Brain-to-Text"
- the later phoneme and sentence stages reuse those source datasets with supervision
- the appendix reports an ablation excluding the human speech datasets from pretraining and says this made no substantial difference for attempted-speech decoding

## Encoder Architecture

### Input Shape And Patchification

The raw neural input has shape:

- `(T, C)`

where:

- `T` is the number of time bins
- `C` is the number of electrodes

The encoder groups every `T_patch` time bins into a time patch. This converts the sequence into:

- `(T / T_patch, C * T_patch)`

The paper explicitly says this follows Feghhi et al. (2025).

The motivation for time patching is:

- speech unfolds on a slower timescale than 20 ms bins
- patching reduces sequence length
- patching reduces redundancy before sending neural representations to an LLM

### Patch Embedding

Each patch is passed through a patch embedding module:

- `LayerNorm -> Linear -> LayerNorm`

This patch embedder converts flattened time patches into transformer token embeddings.

The figure and text also imply subject-specific read-in / read-out layers in the self-supervised setting. The figure caption describes reconstruction through:

- subject-specific linear read-in layers
- subject-specific linear read-out layers

The main text phrases this more generically as:

- patch embedding module on the way in
- reversed patch embedding module on the way out

So the important implementation point is that the model is not purely subject-agnostic at the input/output boundary during pretraining.

### Transformer Core

The neural encoder is a transformer with:

- multi-headed self-attention
- feed-forward blocks
- RoPE relative positional encoding
- bidirectional attention

The paper states that each transformer block contains:

- a self-attention layer
- a feed-forward network

The attention mask is explicitly bidirectional, meaning each patch can attend to all others.

This is a strong offline architecture choice. The discussion section notes that the encoder is therefore not suitable for online decoding without modification.

### Reported Transformer Hyperparameters

Table 10 reports the following encoder hyperparameters:

- embedding dimension: `384`
- head dimension: `512`
- number of heads: `6`
- depth: `7`
- mask ratio: `0.5` for `T12`, `0` for `T15`
- max mask time span: `15`
- patch size: `5`
- dropout rate: `0.2`
- bidirectional: `True`
- attention dropout rate: `0.4`
- white noise standard deviation: `0.2`
- constant offset standard deviation: `0.05`
- Gaussian smoothing width: `2.0`

Important note:

- with `20 ms` bins and patch size `5`, each patch covers `100 ms`

The paper also states:

- the transformer encoder has about `7 million` parameters
- including subject-specific patch embedding modules and linear decoders increases total parameters to about `13 million`

One detail in Table 10 looks unusual:

- reported head dimension `512` is larger than reported embedding dimension `384`

I am recording this as written in the paper rather than correcting it. It may be a typo, a table mismatch, or a nonstandard internal dimension choice.

## Self-Supervised Pretraining Objective

The encoder is pretrained with masked reconstruction on neural time patches.

### Masking Strategy

The paper describes:

- temporal masking inspired by masked autoencoders
- random replacement of some time patches with a learnable mask token
- masked patches may form contiguous spans
- contiguous spans can vary in length up to a maximum timespan
- overall masking ratio is held fixed

### Reconstruction Target

The model reconstructs the original neural activity from partially masked sequences.

The reconstruction loss is:

- MSE

The rationale given is that both threshold crossings and SBP are normalized, making MSE a natural objective.

### Purpose Of Pretraining

The paper claims this stage helps by:

- learning contextual neural representations without labels
- reducing overfitting
- improving robustness to nonstationarity and probe drift
- learning stable representations across species, subjects, tasks, and probe placements

## Phoneme-Decoding Fine-Tuning

After self-supervised pretraining, the encoder is fine-tuned for phoneme decoding.

### Output Head

The transformer outputs are passed through a linear layer that predicts:

- phoneme classes
- blank token
- silence token

The paper later states that they use a `41`-token phoneme vocabulary including:

- all phonemes
- silence
- CTC blank

### Loss

This stage uses:

- CTC loss

### Role In The End-To-End System

A key design choice is that phoneme decoding is not the final output in the end-to-end system. Instead:

- phoneme fine-tuning shapes the encoder representations
- these phoneme-aware latent representations are then fed into the LLM decoder
- phoneme logits themselves are not passed into the LLM

So the phoneme stage acts as an intermediate supervision stage rather than a separate deployed module.

## LLM Decoder Architecture

### Core Interface

The neural encoder outputs are mapped into the LLM embedding space through a shallow MLP projector:

- `Linear -> ReLU -> Linear`

This is the primary interface between neural tokens and language-model tokens.

The model then concatenates:

- projected neural embeddings
- a prompt
- target text embeddings during training

The decoder is trained autoregressively with next-token prediction.

At inference:

- only neural embeddings and the prompt are provided
- the LLM generates text autoregressively

### Prompting

The figure caption gives the example prompt:

- `decode the above neural activity into an English sentence:`

The paper says prompt design varies depending on whether neural embeddings are treated as:

- a distinct neural modality
- an audio modality

### Audio-LLM Versus Text-LLM Modes

The paper evaluates both text-based LLMs and audio-based LLMs.

For text-based LLMs:

- encoder outputs are projected directly into the text embedding space through the shallow MLP

For audio-based LLMs:

- encoder outputs pass through the shallow MLP
- then through the pretrained multimodal projector used by the model’s audio pathway

This creates two possible interpretations of the neural tokens:

- neural modality
- audio modality

The paper reports that treating neural activity as its own neural modality performs slightly better than forcing it into the audio-modality interpretation.

## Modality Aligner And Contrastive Objective

BIT adds a separate modality-alignment module on top of the MLP projector.

### Structure

The modality aligner:

- mean-pools neural embedding tokens
- mean-pools text embedding tokens
- projects them with separate linear layers into a shared latent space
- L2-normalizes the resulting vectors

The paper calls the pooled vectors:

- modality tokens

### Training Objective

The contrastive loss:

- pulls together neural and text embeddings from the same trial / sentence
- pushes apart neural and text embeddings from different examples in the batch

The paper describes this as a symmetric InfoNCE-style objective.

The final sentence-level training loss is:

- cross-entropy loss for token prediction
- plus contrastive loss for alignment

So the final objective is:

- `L_BIT = L_CE + L_contrastive`

This is one of the paper’s main architectural choices: it does not rely on projection alone, but explicitly adds a sentence-level neural-text alignment loss.

## LoRA Fine-Tuning Strategy

The LLM is not fully fine-tuned. Instead, the paper uses LoRA.

LoRA is applied to:

- query projection
- key projection
- value projection
- output projection
- feed-forward projection layers

For audio-based Qwen models, LoRA is also applied to:

- the multimodal projector linear layer

This is a standard parameter-efficient tuning strategy, but it matters because it constrains how much the LLM can adapt to neural input.

### Reported LoRA / End-To-End Hyperparameters

Table 13 reports:

- projector activation: `ReLU`
- learning rate: `5e-5`
- weight decay: `1e-5`
- batch size: `16` for `T12`, `8` for `T15`
- gradient accumulation: `1` for models `< 7B`, `8` for models `>= 7B`
- LoRA rank: `8`
- LoRA scaling factor: `32`
- LoRA dropout: `0.2`

The paper states that all LLM decoders were trained with:

- AdamW
- bfloat16 precision
- `150` epochs

## Alternative Projector Variants Considered

The authors did not only evaluate the MLP projector. They also tested:

- linear projector
- cross-attention projector
- MLP projector

Table 6 reports that the MLP projector performed best on imagined-speech validation WER.

The cross-attention projector is described as:

- neural tokens as queries
- text tokens as keys and values
- LayerNorm and linear projections to a shared hidden dimension
- multi-head cross-attention
- learnable residual connection
- projection back to text embedding dimension

Reported hyperparameters for the cross-attention projector in Table 7:

- hidden dimension: `256`
- number of heads: `1`
- learnable residual scaling factor: `0.5`
- dropout ratio: `0.1`

This matters because it shows the paper explicitly considered more structured neural-text alignment modules but still preferred the simpler MLP.

## Decoder Families Evaluated

The paper benchmarks multiple LLM backbones.

Text-based LLMs mentioned:

- `Qwen2.5-1.5B`
- `Qwen2.5-7B`
- `Qwen3-0.6B`
- `Qwen3-1.7B`

Audio-based LLMs mentioned:

- `Aero1-Audio 1.5B`
- `Qwen2-Audio 7B`

The paper’s headline end-to-end result uses:

- `Aero1-Audio 1.5B`

An important architectural takeaway is that the best-performing end-to-end system did not require the largest LLM. The paper repeatedly emphasizes that smaller audio-LLMs worked well.

## Training Pipeline Summary

The full BIT training pipeline is:

1. pretrain the transformer encoder with masked reconstruction on large cross-species Utah-array data
2. fine-tune the encoder for phoneme decoding with CTC
3. connect the phoneme-aware encoder to an LLM through an MLP projector
4. add a modality aligner for pooled neural-text contrastive alignment
5. fine-tune the encoder + projector + LoRA adapters for sentence-level decoding with cross-entropy plus contrastive loss

This staged procedure is central to the paper. BIT is not trained end-to-end from scratch.

## Cascaded Baseline Used In The Paper

The cascaded version is important because it reveals how the encoder is evaluated independently from the LLM.

The cascaded decoder consists of:

- the pretrained/fine-tuned neural encoder
- CTC phoneme outputs
- a `5`-gram language model with beam search
- an `OPT-6.7B` rescoring model for n-best outputs

This is not the main architecture for end-to-end generation, but it is how they benchmark whether the encoder itself is strong.

## Architectural Strengths Claimed By The Paper

The paper’s argument for this architecture rests on several points:

- transformer encoders may capture longer-range neural structure better than RNNs
- large-scale self-supervised pretraining makes transformers viable despite limited labeled speech data
- phoneme supervision provides a strong intermediate target before sentence-level generation
- a shallow projector is sufficient when paired with a strong pretrained LLM
- explicit contrastive alignment improves neural-text coupling
- audio-LLMs provide a stronger decoder than prior text-only end-to-end approaches

## Architectural Constraints And Limitations Mentioned In The Paper

The paper explicitly notes several limitations tied to the architecture:

- bidirectional attention makes the encoder offline rather than streaming
- causal attention would be needed for online decoding
- LLM decoders remain computationally heavy
- the modality interface may still be improvable
- pretraining depends on large unlabeled datasets

These are not just engineering details; they affect whether the architecture is realistic for deployment versus offline benchmarking.

## Concrete Architectural Choices To Carry Forward

Without endorsing them, the paper defines a useful set of design axes:

- encoder family: RNN versus transformer
- pretraining: none versus SSL
- pretraining corpus: human only versus human + monkey
- input features: threshold crossings only versus threshold crossings + SBP
- tokenization: raw bins versus temporal patches
- patch size
- bidirectional versus causal attention
- phoneme intermediate supervision versus direct sentence supervision
- projector type: linear, MLP, cross-attention
- LLM family: text-only versus audio-LLM
- modality treatment: neural tokens as neural modality versus audio modality
- alignment objective: cross-entropy only versus cross-entropy + contrastive
- adaptation strategy: full fine-tuning versus LoRA

These are likely the most reusable architecture dimensions from the paper for later comparison notes.

## Short Summary

BIT is a staged neural-to-text architecture built around:

- a `7`-layer bidirectional transformer encoder over `100 ms` neural patches
- masked-reconstruction SSL pretraining on cross-species Utah-array data
- phoneme-level CTC fine-tuning
- a shallow MLP projector from neural tokens into LLM embedding space
- a contrastive modality aligner over mean-pooled neural and text embeddings
- an audio-LLM decoder adapted with LoRA for autoregressive sentence generation

The model is architecturally notable less because of any single novel block and more because of the full combination:

- cross-species SSL pretraining
- transformer encoder
- phoneme-aware intermediate supervision
- LLM-based sentence decoding
- explicit contrastive neural-text alignment
