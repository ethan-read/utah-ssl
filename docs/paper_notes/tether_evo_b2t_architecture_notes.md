# Tether Evo Brain2Text Architecture Notes

Paper:

- `Cross-subject decoding of human neural data for speech Brain Computer Interfaces`
- local source: external local paper library, file `tether_evo_b2t.pdf` (not versioned in this repository)

## Scope Of These Notes

These notes summarize the model architecture and adaptation strategy described in the paper, with emphasis on:

- neural input representation
- subject/day adaptation layers
- decoder architecture
- training objective
- decoding pipeline
- architectural choices that may matter for later comparison

These are descriptive notes, not recommendations.

## High-Level Model Structure

The model is a cross-subject neural-to-phoneme decoder with four main parts:

1. a subject- and day-specific affine input transform
2. a shared hierarchical GRU encoder-decoder over time
3. multi-depth phoneme heads trained with hierarchical CTC
4. a phoneme-to-word decoding stage based on WFSTs and a 5-gram LM

The paper's main architectural claim is that cross-subject speech decoding becomes practical when neural activity is first aligned into a shared space with lightweight learned affine transforms.

## Neural Input Representation

For the two main speech datasets:

- neural features are threshold crossings plus spike band power
- both are computed in non-overlapping `20 ms` bins
- the two feature types are concatenated into one neural feature vector per time bin

For the Willett dataset:

- the paper says only premotor electrodes are used, because Broca electrodes were less informative

For the Kunz transfer experiments:

- features are standardized
- trials are padded to a `512`-dimensional representation for downstream modeling

So unlike BIT pretraining, this model does not use a modality-asymmetric regime. It uses the combined `TX + SBP` representation directly.

## Session / Subject Adaptation

### Core Idea

Before the recurrent decoder, each example is passed through a learned affine transform indexed by subject and recording day.

The paper writes this as:

- `x_t_tilde = W_(d,s) x_t + b_(d,s)`

where:

- `s` is subject
- `d` is recording day
- `x_t in R^C` is the neural feature vector at one time step
- `W_(d,s) in R^(C x C)`
- `b_(d,s) in R^C`

This is the most important architectural choice in the paper for transfer-learning purposes.

### Intended Role

The affine transform is meant to:

- compensate for session-to-session drift
- compensate for subject-specific scaling and alignment differences
- map all recordings into a shared latent space before nonlinear decoding

The authors explicitly argue that much of cross-day and cross-subject variation may be correctable with linear realignment rather than a more complex nonlinear adapter.

### What The Paper Claims About The Transforms

The paper treats these transforms as more than simple per-channel rescaling.

The analysis sections argue that they:

- recenter and rotate day-specific neural representations
- reduce day clustering in t-SNE visualizations
- often transfer reasonably well across neighboring days
- capture meaningful geometry related to decoding difficulty

The paper also reports that adapting a pretrained model to new participants by training only the new affine transform can already produce useful decoding performance.

### Architectural Relevance

For our purposes, this paper is strong evidence that lightweight session-conditioned input adaptation is worth testing directly.

## Decoder Architecture

### Overall Structure

After the affine transform, the model processes sequences with a three-block hierarchical GRU decoder.

The blocks are:

- early block: `2` bidirectional GRU layers
- middle block: `2` bidirectional GRU layers
- final block: `1` GRU layer

The hidden size is reported as:

- `d = 2048`

The output vocabulary is:

- phoneme classes plus the CTC blank token

### Important Causality Note

The paper sometimes describes the model as a causal decoder in high-level prose.

However, the method section explicitly states that:

- the first two blocks are bidirectional GRUs

So the implemented model is not strictly causal in the online-decoding sense.

That discrepancy matters for later comparison with genuinely causal SSM or streaming models.

## Hierarchical Phoneme Feedback Design

The paper's main decoder novelty is not the recurrent backbone itself, but the way phoneme predictions are fed back between blocks.

### Early Block

The early GRU block produces hidden states:

- `z1 = GRU_early(X_tilde)`

An auxiliary classifier maps these hidden states to phoneme logits:

- `l1 = W_early z1 + b_early`

Softmax produces phoneme probabilities:

- `p1 = Softmax(l1)`

These probabilities are then projected back to the hidden dimension:

- `p1_hat = W_proj,1 p1 + b_proj,1`

This feedback term is added to the early hidden states:

- `h1 = z1 + p1_hat`

### Middle Block

The middle block repeats the same pattern:

- `z2 = GRU_middle(h1)`
- `l2 = W_middle z2 + b_middle`
- `p2 = Softmax(l2)`
- `p2_hat = W_proj,2 p2 + b_proj,2`
- `h2 = z2 + p2_hat`

### Final Block

The final block is a single GRU layer:

- `z3 = GRU_final(h2)`

followed by the final phoneme classifier:

- `l3 = W_final z3 + b_final`

### Architectural Interpretation

The intent is to partially mitigate the conditional-independence limitation of standard CTC by letting deeper recurrent layers see earlier phoneme hypotheses.

So the model remains CTC-based, but it injects a limited form of autoregressive-like conditioning through internal feedback rather than through a fully autoregressive decoder.

## Training Objective

### Hierarchical CTC

Training uses CTC at all three prediction depths.

The total loss is:

- `L_total = L_CTC(l3, y) + lambda * L_CTC(l2, y) + L_CTC(l1, y)`

The reported setting is:

- `lambda = 0.3`

So the auxiliary heads are not only for analysis; they are active supervision points during training.

### Motivation

The paper frames hierarchical CTC as a compromise:

- keep the stability and alignment-free training of CTC
- reduce some of CTC's independence limitations
- avoid the instability they associate with autoregressive transformer training in this domain

## Training Details

Reported training details include:

- optimizer: `Adam`
- batch size: `64`
- training length: `120k` steps
- learning rate warmup: `0 -> 5e-3` over first `1k` steps
- learning rate schedule: cosine decay to `1e-4` by step `120k`
- weight decay: `1e-5`
- mixed precision training
- gradient accumulation
- Gaussian noise augmentation
- small per-channel offset augmentation

The paper says the hyperparameters were largely matched to the original Card baseline, except that model dimensionality was increased to `d = 2048`.

## Output Decoding

The neural model predicts phoneme logits rather than text directly.

Sentence decoding is done with:

- WFST decoding
- pronunciation lexicon
- `5`-gram language model
- beam search

Optional rescoring with a larger pretrained LM such as OPT is mentioned.

So this remains a cascaded phoneme-first architecture rather than an end-to-end neural text generator.

## Transfer / Adaptation Regimes

The paper evaluates several adaptation settings for new participants:

- train only the new subject/day affine transforms
- fine-tune the whole pretrained model
- train from scratch on the target participant

This is a useful experimental template because it directly separates:

- lightweight alignment-only transfer
- full-parameter transfer
- no-transfer baseline

For our work, this is one of the most reusable experimental ideas in the paper.

## Main Architectural Choices To Carry Forward

Without endorsing them, the paper defines several design axes that are relevant for our own model design:

- combine `TX` and `SBP` as one per-timestep feature vector
- use explicit session / subject-specific affine alignment before the shared backbone
- keep the shared backbone relatively simple
- supervise intermediate layers, not only the final output
- compare lightweight adaptation against full fine-tuning
- treat domain adaptation as an input-alignment problem rather than only a backbone problem

## Main Architectural Differences From The Other Papers

Relative to BIT:

- no transformer encoder
- no self-supervised pretraining
- no LLM decoder
- much more explicit session/day adaptation
- fully phoneme-first rather than neural-to-text generation

Relative to Cortical-SSM:

- recurrent speech decoder rather than SSM
- no explicit structured frequency front end
- strong domain-shift handling through affine transforms

Relative to POSSM:

- fixed binned inputs rather than individual spike tokenization
- offline recurrent decoding rather than a streaming causal design
- explicit linear alignment per day / subject

## Practical Takeaways For Our Design Discussion

The most relevant contribution for our purposes is the learned affine alignment layer.

This paper provides concrete support for the idea that:

- simple learned linear transforms can materially help cross-session and cross-subject transfer
- adaptation does not necessarily require a large session-specific subnet
- a useful baseline should include explicit session-aware input adaptation

At the same time, the paper is less informative for our intended causal SSL setup because:

- the reported decoder is not strictly causal in implementation
- the training objective is supervised hierarchical CTC rather than self-supervised pretraining
