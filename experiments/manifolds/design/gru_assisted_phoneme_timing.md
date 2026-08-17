# GRU-assisted phoneme timing for raw-bin analyses

Status: design considerations only. No new experiment has been implemented or
validated from this note.

## Core distinction

**Repository observation:** the local Willett-style GRU consumes overlapping
patches of 14 neural bins (`280 ms`) with a stride of four bins (`80 ms`). One
patch produces one recurrent hidden state and one phoneme-logit vector. Native
GRU outputs therefore occur every 80 ms, although the underlying neural data
are stored in 20 ms bins. Adjacent patches share 10 of 14 bins.

The existing manifold work correctly exported these internal GRU states and
fit PCA to them. This measures decoder-state geometry. It should not be
described as PCA of instantaneous 20 ms neural population activity.

## Using the GRU as a timing annotator

**Adapted proposal:** a frozen GRU can provide model-assisted phoneme timing
when no independent timestamps exist. Force-align the known transcript to its
CTC logits, then map the resulting 80 ms token posteriors onto the original
20 ms grid and use them to select or weight raw SBP bins.

Prefer a CTC forward-backward posterior over a single hard Viterbi path. The
posterior preserves boundary uncertainty and supports timing-jitter analyses.
A per-bin label produced by interpolation or repetition has 20 ms grid spacing,
but not genuine 20 ms temporal resolution.

Running the checkpoint with stride one is mechanically possible but changes
the recurrent update interval from its trained 80 ms cadence to 20 ms. Treat
that as out-of-distribution unless the model is retrained or calibrated. Four
phase-shifted stride-four passes can cover offsets 0, 1, 2, and 3 bins while
preserving the trained cadence, but they form four distinct recurrent
trajectories and are best used as a timing-sensitivity check.

## Circularity and required controls

GRU-derived timing depends on the neural signal being analyzed. It is therefore
not independent evidence of neural phoneme timing. Using it to probe raw SBP is
still more informative than using it to select and then probe the same GRU
hidden states, provided the limitation and controls are explicit.

Minimum controls:

- generate timings with a frozen model that did not train on the evaluation
  session, where practical;
- fit normalization, PCA, and alignment only on training sessions;
- evaluate geometry or decoding on held-out sessions and held-out prompts;
- compare hard paths with posterior-weighted and timing-jittered analyses;
- include matched within-utterance random centers and shuffled correspondence;
- report the GRU checkpoint, patch size, stride, feature mode, and timing-anchor
  convention exactly.

## Relationship to Spalding et al.

**Reported:** Spalding et al. fit patient-specific PCA directly over the neural
channel dimension, retained enough components to explain 90% of input variance,
and used condition- and time-matched CCA to align patients. See the paper notes
at [`docs/paper_notes/spalding_shared_latent_speech_alignment_notes.md`](../../../docs/paper_notes/spalding_shared_latent_speech_alignment_notes.md).

**Adapted proposal:** for a paper-like Utah analysis, fit separate PCA bases to
instantaneous 128-channel SBP vectors at the original 20 ms cadence. Use the
GRU only to supply soft phoneme/relative-time correspondence. Compare the 90%
variance rule with fixed and nested-selected dimensions, because high-variance
directions may encode session drift rather than transferable speech structure.

### Raw-bin normalization caveat

**Reported:** the paper describes PCA of baseline-subtracted high-gamma
activity and does not report per-channel z-scoring before PCA.

**Repository observation:** the first raw-20-ms Utah notebook compares
session-wise z-scoring with training-global z-scoring. In that notebook,
"raw" refers to the native 20 ms SBP bins rather than unstandardized channel
amplitudes. Both implemented conditions estimate PCA from a correlation-like
matrix, so they can produce a substantially flatter spectrum than covariance
PCA on centered SBP amplitudes.

**Adapted follow-up:** retain a centered-only, training-mean PCA as a separate
paper-like sensitivity condition before comparing Utah component counts with
the paper. Do not replace the z-scored conditions: session-wise z-scoring asks
about within-session geometry after removing day-specific offsets and gains,
while training-global z-scoring tests future-session drift. The centered-only
condition asks a different question and may be dominated by a few high-variance
channels. Likewise, treat the earlier top-six variance from flattened 280 ms
decoder windows as descriptive context only because its dimensionality,
sampling, and temporal mixing differ from instantaneous-bin PCA.

| Element | Paper | Proposed Utah adaptation | Main uncertainty |
|---|---|---|---|
| Neural PCA | Per-patient channel activity | Per-subject 20 ms SBP vectors | Session variance may dominate |
| Timing | Measured response onset and repeated conditions | Cross-fitted GRU CTC posterior | Model-assisted and 80 ms native cadence |
| Alignment | Condition/time-matched CCA | Phoneme/relative-time CCA | Coarticulation and duration mismatch |
| Endpoint | Held-out target decoding | Held-out-session phoneme probe or PER | Shared-prompt leakage must be excluded |

## Open decisions before implementation

1. Which frozen checkpoint supplies timing, and which sessions trained it?
2. Hard Viterbi spans or CTC forward-backward posteriors?
3. Patch-center, patch-end, or empirically calibrated token timestamps?
4. Whether four phase-shifted passes materially stabilize boundaries?
5. PCA dimension rule and regularized-CCA strength?
6. Primary split: held-out sessions, held-out prompts, or both?

The older `MODEL_KEY="gru"` trajectory export has incomplete checkpoint
provenance and must not be conflated with `gru_released`. The local GRU code is
an LLM-assisted port/adaptation with unresolved upstream provenance; see
[`experiments/supervised_baselines/PROVENANCE.md`](../../supervised_baselines/PROVENANCE.md).

## AI assistance

Codex drafted and later updated this design note by synthesizing the repository
code, saved notebook behavior, the local Spalding paper, and the accompanying
paper notes. No analytical result was produced. Human review is required before
these choices become an experiment specification.
