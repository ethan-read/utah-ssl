# Objective

This folder is for active experiments that test whether generic
self-supervised learning improves SSM-based neural speech decoding on Utah
array data.

# Current Research Direction

The main goal is to build clean, reusable SSL experiments around generic `S5`
and `Mamba` encoders, then evaluate whether SSL pretraining improves downstream
Brain2Text24 phoneme CTC decoding compared with the same architectures trained
from random initialization.

The preferred next direction is generic masked neural modeling rather than
POSSM-specific reconstruction:

- compare raw-bin, causal temporal patch, and light causal conv-stem inputs
- test `S5` and `Mamba` under matched downstream decoding protocols
- use downstream CTC fine-tuning/probing as the decisive metric
- treat SSL loss, retrieval accuracy, and reconstruction error as diagnostics,
  not as sufficient evidence of useful speech representations

# Reference Baselines

The Willett-style supervised CTC code is the strongest recipe baseline and
should remain the anchor for splits, normalization, smoothing, temporal
patching, and decoder evaluation.

The POSSM code is now a reference implementation and evidence source, not the
primary proof vehicle. POSSM results show that reconstruction pretraining can
help over random initialization, but POSSM-specific interfaces have not closed
the gap to supervised Willett-style SSM decoding.

# Cleanup Direction

Prefer a smaller experiment base with clear reusable modules over many
one-off notebooks and launch scripts. Completed sweeps, compatibility wrappers,
and simple scripts can be deleted when their conclusions have been recorded in
`docs/notes/experiment_synthesis.md` or archived notes.

Keep cache migration, normalization-stat, and data-integrity utilities when
they protect reproducibility or are referenced by tests/notebooks.

When running cache-prep or corpus-audit scripts, prefer the full local cache
root at `/Users/home/thesis/data/cache_v1`. The repo-local cache under
`/Users/home/thesis/utah-ssl/data/cache_v1` can be a smaller working subset and
should not be treated as the default full-corpus BIT cache without checking its
contents first.
