# BIT-to-S5 Codebase Checklist

This document translates the abstract BIT-to-`S5` adaptation plan into concrete work against the current repository.

It is written as an implementation checklist, not a paper summary.

The main question is:

- given the code that already exists here, what needs to be reused, changed, or added to build a faithful BIT-style `S5` system?

## Notebook Cross-Check

I also checked:

- [`analysis/active/ssl_experiments/archive/notebooks/s5_maskedreconstruction.ipynb`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/archive/notebooks/s5_maskedreconstruction.ipynb)

That notebook is important because it shows how the current masked-reconstruction path is actually being used in practice.

What the notebook currently implements:

- causal `S5` masked reconstruction
- `TX`-only pretraining
- patch-level masking by default
- default `patch_size = 4`
- default `patch_stride = 2`
- segment length `80` bins
- held-out `Brain2Text25` downstream probe workflow
- frozen linear-probe comparison as the main apples-to-apples downstream test
- optional full fine-tuning downstream diagnostic

What that means for BIT fidelity:

- it is a strong stage-1 prototype
- but it is not yet a faithful BIT port, because it is still:
  - causal instead of offline bidirectional
  - configured around `patch_size = 4`, `patch_stride = 2` instead of BIT's default `5`
  - centered on frozen probe transfer rather than a true phoneme fine-tuning stage
  - missing the sentence-level `LLM` stage entirely

## Short Answer

The repo already has strong pieces for the first half of the adaptation:

- a pure-PyTorch `S5` reference backbone
- patch-based `S5` SSL models
- session-keyed read-in / read-out banks
- masked reconstruction machinery
- held-out phoneme probe infrastructure
- canonical Brain2Text25 cache / manifest handling

The repo does not yet have the full BIT stack:

- no offline bidirectional `S5`
- no BIT-style phoneme fine-tuning stage as the primary training pipeline
- no cascaded `5`-gram / `OPT` evaluation path inside the active benchmark code
- no sentence-level `LLM` stage
- no `LoRA`
- no BIT-style neural-text contrastive sentence alignment

So the current codebase is already close to:

- BIT stage 1: SSL masked reconstruction

and partially close to:

- BIT stage 2: phoneme fine-tuning

but still far from:

- BIT stage 3: end-to-end sentence generation with `LLM + LoRA + contrastive alignment`

## Best Existing Building Blocks

### 1. Canonical `S5` Backbone

Best file:

- [`analysis/active/transfer_benchmark/ssl_autoresearch/s5.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/s5.py:1)

What it already gives you:

- a clean pure-PyTorch `S5SequenceBackbone`
- a stable reference `S5Block`
- explicit sequence-length masking support

How to use it:

- keep this as the shared low-level `S5` implementation
- do not fork another incompatible `S5` block unless a real limitation forces it

### 2. BIT-Like SSL Encoder Skeleton

Best file:

- [`analysis/active/ssl_experiments/masked_ssl/model.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/model.py:1)

What it already gives you:

- patchification over neural sequences
- patch embedding with `LayerNorm -> Linear -> LayerNorm`
- `S5MaskedEncoder`
- session-keyed read-in bank
- reconstruction head path
- session-keyed read-out bank

Why this matters:

- this is already the closest thing in the repo to the BIT encoder boundary structure
- it is much closer to BIT than the transfer benchmark’s current future-prediction encoder

### 3. Masked Reconstruction Objective

Best file:

- [`analysis/active/ssl_experiments/masked_ssl/objectives.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/objectives.py:1)

What it already gives you:

- patch-level masking
- contiguous masked spans
- masked-only reconstruction loss
- support for raw continuous patch reconstruction targets

Why this matters:

- this is the strongest existing fit to BIT stage 1

### 4. SSL Training Loop

Best file:

- [`analysis/active/ssl_experiments/masked_ssl/training.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/training.py:1)

What it already gives you:

- a checkpointed SSL training loop
- source-session discovery
- feature-mode configuration
- masking configuration

Why this matters:

- this is the best existing home for a faithful BIT-style stage-1 implementation

### 5. Held-Out Phoneme Probe Infrastructure

Best files:

- [`analysis/active/ssl_experiments/masked_ssl/probe.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/probe.py:1)
- [`analysis/active/transfer_benchmark/ssl_autoresearch/train.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/train.py:1)

What they already give you:

- linear `CTC` probe heads
- held-out session evaluation
- canonical Brain2Text25 data loading
- adaptation-regime scaffolding

Why this matters:

- these files are the natural base for BIT stage 2

## Current Mismatches With Faithful BIT

### 1. The Active Benchmark Is Causal, Not BIT-Like Offline

Current files:

- [`analysis/active/transfer_benchmark/ssl_autoresearch/train.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/train.py:1)
- [`analysis/active/ssl_experiments/masked_ssl/model.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/model.py:1)

Mismatch:

- BIT’s encoder is explicitly bidirectional and offline
- current `S5` usage here is causal / forward-only

What to do:

- add one offline bidirectional `S5` wrapper before claiming BIT faithfulness

### 2. The Benchmark Objective Is Still Future Prediction

Current file:

- [`analysis/active/transfer_benchmark/ssl_autoresearch/train.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/train.py:1)

Mismatch:

- the active benchmark loop is wired to `future_prediction`
- BIT stage 1 uses masked reconstruction, not future prediction

What to do:

- do not try to force BIT into the current `future_prediction` benchmark loop
- instead, treat `masked_ssl` as the stage-1 BIT path and only reuse benchmark data / probe pieces where helpful

### 3. Subject-Specific Layer Placement Differs Across Modules

Current code paths:

- `masked_ssl/model.py`: session-keyed read-in on patched tokens plus session-keyed read-out
- `ssl_autoresearch/train.py`: session affine at raw input level before patching

Mismatch:

- BIT evidence points most directly to subject-specific read-in / read-out around the patch-token interface
- the benchmark scaffold currently encodes adaptation as a raw-feature affine bank

What to do:

- for the faithful BIT path, prefer the `masked_ssl` style boundary layers
- do not let the transfer-benchmark affine design silently redefine the BIT adaptation mechanism

### 4. Patching Semantics Differ

Current code paths:

- `masked_ssl/model.py` appends a terminal patch if needed
- `ssl_autoresearch/train.py` intentionally drops trailing bins that do not fit the stride schedule

Mismatch:

- a faithful BIT reproduction should use one consistent patching rule everywhere

What to do:

- choose one BIT-like rule and apply it consistently across SSL and phoneme fine-tuning
- the simplest faithful choice is:
  - `patch_size = 5`
  - `patch_stride = 5`
  - non-overlapping patches

### 5. No Sentence-Level LLM Stack Exists Yet

Current state:

- no sentence projector into `LLM` embeddings
- no prompt handling
- no modality aligner
- no `LoRA`
- no token-level autoregressive sentence decoder

Mismatch:

- this is the entire BIT stage 3

What to do:

- treat this as a new implementation phase, not a small refactor

## Recommended Implementation Home

If the goal is a faithful BIT-style path, the cleanest code organization is:

- keep low-level `S5` blocks in:
  - [`analysis/active/transfer_benchmark/ssl_autoresearch/s5.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/s5.py:1)
- keep BIT-style SSL encoder / masked reconstruction path in:
  - [`analysis/active/ssl_experiments/masked_ssl/`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl)
- add a new BIT-style phoneme fine-tuning module alongside it
- add a new sentence-level module separately rather than cramming it into the current transfer benchmark

Why:

- the active benchmark loop is currently optimized around causal future-prediction experiments
- BIT fidelity pulls in a different training order and a different stage structure

## Concrete Checklist

## Phase 1: Stabilize The Shared Encoder Surface

### 1. Add One Shared Bidirectional `S5` Wrapper

Likely file:

- [`analysis/active/transfer_benchmark/ssl_autoresearch/s5.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/s5.py:1)

Work:

- add a `BiS5SequenceBackbone` or equivalent wrapper
- support:
  - forward `S5`
  - backward `S5`
  - merge by concat or sum
- preserve length masking

Completion criterion:

- one sequence-in / sequence-out offline `S5` module usable by both SSL and phoneme fine-tuning

### 2. Standardize One Patch Policy

Likely files:

- [`analysis/active/ssl_experiments/masked_ssl/model.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/model.py:1)
- [`analysis/active/transfer_benchmark/ssl_autoresearch/train.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/train.py:1)

Work:

- choose one patch builder behavior
- use the same behavior in:
  - SSL stage
  - phoneme fine-tuning stage
  - any future sentence stage

Faithful default:

- `patch_size = 5`
- `patch_stride = 5`
- non-overlapping temporal patches

Completion criterion:

- no disagreement between SSL and downstream about tokenization semantics

### 3. Standardize The Boundary Adaptation Mechanism

Likely files:

- [`analysis/active/ssl_experiments/masked_ssl/model.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/model.py:1)
- [`analysis/active/transfer_benchmark/ssl_autoresearch/train.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/train.py:1)

Work:

- choose the BIT-faithful default:
  - subject/session-keyed read-in around patch tokens
  - subject/session-keyed read-out for SSL reconstruction
- keep the shared `S5` core subject-agnostic

Completion criterion:

- a single encoder family exposes the same subject/session conditioning across stages

## Phase 2: Make Stage 1 Truly BIT-Like

### 4. Keep `masked_ssl` As The Main Stage-1 Path

Primary files:

- [`analysis/active/ssl_experiments/masked_ssl/model.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/model.py:1)
- [`analysis/active/ssl_experiments/masked_ssl/objectives.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/objectives.py:1)
- [`analysis/active/ssl_experiments/masked_ssl/training.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/training.py:1)

Work:

- switch the backbone from causal `S5` to offline bidirectional `S5`
- keep:
  - masked patch reconstruction
  - `MSE`
  - session-keyed read-in / read-out
- make `patch_size = 5` the BIT-faithful default configuration

Completion criterion:

- a checkpointed SSL run that is much closer to BIT stage 1 than the future-prediction scaffold

### 5. Keep The Pretraining Feature Policy Explicit

Primary files:

- [`analysis/active/ssl_experiments/masked_ssl/training.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/training.py:1)
- cache / sampler code under [`analysis/active/ssl_experiments/masked_ssl/`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl)

Work:

- keep `feature_mode = tx_only` as the BIT-faithful pretraining default
- do not silently move to `tx_sbp` pretraining unless that is an intentional non-faithful variant

Completion criterion:

- stage-1 checkpoints carry an explicit BIT-style feature policy

## Phase 3: Build A Real Phoneme Fine-Tuning Stage

### 6. Create A Dedicated BIT-Style Phoneme Fine-Tuning Module

Recommended new file:

- `analysis/active/ssl_experiments/masked_ssl/phoneme_finetune.py`

Why a new file:

- the current `probe.py` is set up for cheap held-out linear probes
- BIT stage 2 is stronger than a cheap probe:
  - it is the actual supervised fine-tuning stage used before sentence decoding

Work:

- load a stage-1 SSL checkpoint
- keep the pretrained encoder weights
- remove the SSL reconstruction head
- attach a phoneme linear head
- train with `CTC`
- support:
  - frozen-encoder probe mode
  - full encoder fine-tuning mode
- make full phoneme fine-tuning the default BIT-faithful path

Completion criterion:

- a saved phoneme-tuned encoder checkpoint usable for both cascaded evaluation and later sentence decoding

### 7. Reuse Canonical Probe Data, But Not Probe Assumptions

Best reusable files:

- [`analysis/active/ssl_experiments/masked_ssl/probe.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/probe.py:1)
- [`analysis/active/transfer_benchmark/ssl_autoresearch/data.py`](/Users/home/thesis/utah-ssl/analysis/active/transfer_benchmark/ssl_autoresearch/data.py:1)

Work:

- reuse:
  - canonical Brain2Text25 manifest reading
  - session splitting
  - phoneme target collation
- do not keep the “cheap probe only” framing for the BIT path

Completion criterion:

- stage 2 runs on the same data surfaces as the benchmark, but as real fine-tuning rather than only probing

### 8. Support Downstream `TX + SBP`

Current mismatch:

- stage-1 code often assumes `tx_only`

Work:

- make stage 2 explicitly support `tx_sbp`
- check that input dimensionality and feature masking stay aligned when switching from `tx_only` SSL to `tx_sbp` phoneme fine-tuning

Completion criterion:

- phoneme fine-tuning can faithfully match BIT’s downstream feature policy

## Phase 4: Add The Cascaded Evaluation Path

### 9. Implement A Cascaded Decoder Entry Point

Recommended new file:

- `analysis/active/ssl_experiments/masked_ssl/cascaded_decode.py`

Work:

- accept phoneme logits or phoneme model outputs
- plug into the benchmark-style external decoding resources
- support:
  - `5`-gram decoding
  - optional `OPT` rescoring later

Completion criterion:

- one reproducible cascaded evaluation path for the phoneme-tuned `S5` encoder

Important note:

- if the external `5`-gram / `OPT` resources are not locally wired yet, the code can start with a clear stub boundary
- but keep the interface explicit so the BIT comparison is not lost

## Phase 5: Add The Sentence-Level BIT Stack

### 10. Create A Sentence-Stage Module Instead Of Extending The Probe Code

Recommended new package:

- `analysis/active/ssl_experiments/bit_sentence/`

Suggested files:

- `model.py`
- `training.py`
- `data.py`
- `lora.py`
- `alignment.py`

Why:

- this stage is architecturally different enough that it deserves its own surface

### 11. Implement The Neural-to-LLM Projector

Recommended file:

- `analysis/active/ssl_experiments/bit_sentence/model.py`

Work:

- take the phoneme-tuned `S5` hidden-state sequence
- apply the default BIT projector:
  - `Linear -> ReLU -> Linear`
- return neural embeddings in the `LLM` token space

Completion criterion:

- a clean module boundary between the `S5` encoder and the `LLM`

### 12. Implement The Modality Aligner

Recommended file:

- `analysis/active/ssl_experiments/bit_sentence/alignment.py`

Work:

- mean-pool neural token embeddings
- mean-pool text token embeddings
- map both through modality-specific linear heads
- `L2` normalize
- compute symmetric `InfoNCE`

Completion criterion:

- sentence-level loss can be written as `L_CE + L_contrastive`

### 13. Add Prompt Handling

Recommended file:

- `analysis/active/ssl_experiments/bit_sentence/model.py`

Work:

- support the BIT-style prompt templates
- start with the neural-modality prompt

Completion criterion:

- neural token sequence plus prompt can be assembled deterministically for `LLM` fine-tuning

### 14. Add `LoRA`

Recommended files:

- `analysis/active/ssl_experiments/bit_sentence/lora.py`
- or direct integration inside the sentence model package

Work:

- adapt the target `LLM` through `LoRA`, not full fine-tuning
- keep the encoder, projector, aligner, and `LoRA` adapters trainable

Completion criterion:

- the sentence stage follows BIT’s parameter-efficient fine-tuning pattern

## Phase 6: Write The Experiment Contract Clearly

### 15. Add A Dedicated README For The BIT-Style Path

Recommended file:

- `analysis/active/ssl_experiments/masked_ssl/README_BIT_path.md`

Work:

- document the three stages:
  - SSL masked reconstruction
  - phoneme `CTC` fine-tuning
  - sentence-level `LLM` fine-tuning
- document which checkpoints feed the next stage

Completion criterion:

- one place in the repo explains how to run the full BIT-style `S5` pipeline

## Already Existing Tests Worth Keeping

Best file:

- [`analysis/active/ssl_experiments/masked_ssl/tests/test_masked_ssl.py`](/Users/home/thesis/utah-ssl/analysis/active/ssl_experiments/masked_ssl/tests/test_masked_ssl.py:1)

Why it matters:

- it already checks:
  - patching
  - session-keyed read-in / read-out routing
  - masking behavior
  - target-affine gradient paths

What to add later:

- bidirectional `S5` shape tests
- phoneme fine-tuning checkpoint handoff tests
- `tx_only` to `tx_sbp` stage transition tests
- sentence-stage projector / aligner shape tests

## Recommended Order Of Work

If the goal is fastest progress with the least wasted refactoring, do the work in this order:

1. add offline bidirectional `S5`
2. standardize one patching policy
3. keep `masked_ssl` as the BIT-style stage-1 home
4. create a dedicated phoneme fine-tuning stage instead of overloading the probe path
5. wire in cascaded evaluation
6. only then add the sentence-level `LLM + LoRA + contrastive` stage

Do not start with the `LLM` stage first.

The current repo is already closest to BIT at the SSL stage, so the cleanest path is to finish the encoder-side pipeline before adding the largest missing component.

## Minimum Viable Faithful Path In This Repo

A minimum viable faithful BIT-style `S5` implementation in this repo would look like:

- stage 1:
  - `masked_ssl/model.py`
  - `masked_ssl/objectives.py`
  - `masked_ssl/training.py`
  - with bidirectional `S5`, `patch_size = 5`, `tx_only`, subject/session-keyed read-in/read-out
- stage 2:
  - new `masked_ssl/phoneme_finetune.py`
  - `tx_sbp`
  - linear phoneme head with `CTC`
  - saved phoneme-tuned encoder checkpoint
- stage 2.5:
  - new `masked_ssl/cascaded_decode.py`
  - optional `5`-gram / `OPT` path
- stage 3:
  - new `bit_sentence/` package
  - `MLP` projector
  - modality aligner
  - `LoRA`
  - sentence-level `L_CE + L_contrastive`

If one of those stages is missing, call it:

- BIT-inspired `S5`

not:

- faithful BIT-to-`S5`

## Bottom Line

The codebase already contains the right foundation for a faithful BIT-to-`S5` path, but it is split across two worlds:

- `masked_ssl` contains the best BIT-like SSL machinery
- `ssl_autoresearch` contains the best benchmark/data plumbing

The clean implementation strategy is:

- use `masked_ssl` as the model-development home for stages 1 and 2
- reuse data and evaluation infrastructure from `ssl_autoresearch`
- add the sentence-level stack as a new dedicated module

That keeps the BIT adaptation legible and avoids forcing it into the current causal future-prediction benchmark structure.
