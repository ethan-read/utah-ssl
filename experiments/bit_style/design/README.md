# Current BIT-style design

This branch tests whether BIT-style masked reconstruction on a broad Utah-array
corpus improves subsequent Brain-to-Text phoneme decoding when the Transformer
backbone is replaced with S5. It is a paper-inspired adaptation, not an
official BIT implementation.

## Scientific contract

Stage 1 uses TX because most auxiliary datasets do not provide SBP. Dataset
membership and source splits are defined positively by
`utah_ssl.bit_cache_contract.BIT_STAGE1_DATASET_SPLITS`. Brain-to-Text 2024 and
2025 contribute only their first 128 area-6v TX channels; the shared model input
may zero-pad narrower physical inputs.

The downstream Brain-to-Text default is 128-channel area-6v SBP loaded from the
validated clipped-FP16 cache. TX and TX+SBP are explicit modality comparisons;
TX+SBP is particularly relevant when measuring fidelity to the BIT paper, but
it is not the repository default.

The primary transfer test is phoneme CTC, compared against the same architecture
trained from random initialization. Reconstruction loss alone is not evidence
of useful transfer.

## Current implementation

The branch currently provides:

- broad multi-dataset TX sampling with explicit dataset and signal contracts;
- temporal patching, causal-convolution, and raw-bin input modes;
- causal or bidirectional S5 backbones, plus a causal Mamba comparison;
- masked time/channel reconstruction;
- downstream CTC training and checkpoint transfer;
- tests and Modal launchers for the current Stage-1/Stage-2 path.

The current CLI defaults (`causal`, 14-bin patches, 4-bin stride) are an
implementation baseline, not a claim of BIT fidelity. The detailed controlled
backbone-substitution recipe remains in
[`faithful_bit_to_s5_adaptation.md`](faithful_bit_to_s5_adaptation.md).

## Known gap

The current generic configuration carries one `SignalSpec` across Stage 1 and
Stage 2. A canonical broad-TX-to-Brain-to-Text-SBP transfer experiment therefore
still needs an explicit downstream signal handoff and matching normalization
artifact validation. Until that is implemented, the branch should not be
described as having completed the intended BIT transfer test.

The cascaded language-model decoder, sentence-level LLM projector, LoRA, and
neural-text contrastive alignment described by BIT are not implemented here.
They should remain secondary until the phoneme-transfer comparison establishes
that Stage-1 pretraining is useful.

## Evidence

Canonical results belong in [`../results/`](../results/). The old codebase audit
is preserved under
`experiments/archive/generic_ssm_ssl/design/bit_to_s5_codebase_audit.md`.
