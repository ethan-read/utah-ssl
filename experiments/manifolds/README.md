# Neural Trajectories in Attempted Speech

This active branch groups exploratory representation-manifold and neural-
trajectory analyses. Canonical interpretations live under `results/`; notebooks
remain executable evidence and exploration surfaces.

This folder tests whether phoneme-related population paths are repeatable. It
starts from the durable GRU representation export inspected by
`notebooks/export_willett_representations.ipynb` and keeps full per-utterance
time order, which the pooled PCA plots discard.

## Coarse articulatory activity

The versioned [articulatory feature taxonomy](design/articulatory_feature_taxonomy.md)
defines the canonical transcript-derived movement targets and their
machine-readable 41-token record.

`notebooks/coarse_articulatory_activity.ipynb` runs the best recorded local GRU
on all 24 validation sessions. It measures on the final four how much native
phoneme error remains after mapping predictions to broad and manner groups,
then fits independent lips, tongue-front, and tongue-body linear probes of
reference-aligned GRU hidden states. Probe heads train on the first 20 sessions
and evaluate the final four; the frozen GRU itself is not session-held-out.
These accessibility tests determine whether coarse-label retraining is
warranted before the decoder-assisted raw-SBP analysis.

## Layerwise GRU export

`scripts/export_gru_layer_states.py` reruns the checkpoint used by the
articulator probe and saves the complete 80 ms hidden-state sequence from every
GRU layer. It clones the stored recurrent weights into an evaluation-only
one-layer stack and checks the reconstructed top layer and logits against the
ordinary model forward pass on every batch before saving. The default full
artifact is written beneath:

```text
/content/drive/MyDrive/utah_ssl/data/representations/willett_manifolds/
  gru_layerwise_b2t24_step18300_v1/gru_best_step18300_all_val_sessions
```

After mounting Drive in Colab, run a separate smoke export first:

```bash
python -m experiments.manifolds.scripts.export_gru_layer_states --smoke
```

Then run the complete 880-example validation export:

```bash
python -m experiments.manifolds.scripts.export_gru_layer_states
```

The script refuses to reuse an existing destination unless `--overwrite` is
explicit. Intermediate states default to FP16 on disk, while numerical
equivalence is checked in the model compute dtype before casting. It writes to
a sibling staging directory, reopens and validates every shard (including the
stored final-layer values), writes `validation.json` and `_SUCCESS.json`, and
only then renames the completed artifact into the canonical destination.

## What the 14-bin GRU input does and does not mean

The Willett model consumes a causal sequence of overlapping 280 ms patches
(`14 x 20 ms`) every 80 ms (`stride=4`). Its hidden states therefore do form a
time-ordered, causal representational trajectory. They are useful for asking
what dynamics support this decoder. They are not a clean measurement of
sub-80-ms neural dynamics, and a path in this space is not by itself evidence
for an intrinsic biological dynamical system.

For a neural claim, compare three levels:

1. normalized raw activity at the original 20 ms cadence (primary control;
   this still needs a raw-bin export),
2. exported `input_windows` or `adapted_input_windows` at 80 ms cadence, and
3. GRU/S5 `hidden` states at 80 ms cadence.

The existing input-window control is a flattened overlapping 280 ms window. It
is closer to the neural measurements than the hidden state, but it is still
temporally mixed. Do not call it raw-bin dynamics.

## Raw-SBP bigram transition trajectories

`notebooks/raw_bigram_transition_pca.ipynb` uses the complete step-18,300 GRU
logit export only for transcript-constrained CTC timing, then extracts native
20 ms trajectories from the canonical clipped-FP16 128-channel SBP cache. It
fits equal-bigram-weighted change and state PCA views for the 66 transcript
bigrams occurring at least 50 times, ranks mean paths by top-six trajectory
captured fraction, and reports session repeatability and +/-40 ms timing
sensitivity. All 24 validation/model-selection sessions participate, so this
is explicitly descriptive and transductive rather than a held-out-session
result.

## First experiment

The runner:

- Viterbi-aligns the known reference phoneme sequence to CTC logits;
- extracts a fixed window around each aligned phoneme occurrence;
- fits one shared standardized PCA basis across all retained trials;
- plots individual paths and condition means;
- estimates per-phoneme split-half reliability; and
- tests whether paths are closer within phoneme than between phonemes using a
  label permutation test.

The two repeatability metrics subtract each path's temporal mean first. This
prevents a stationary phoneme cluster from counting as repeatable motion.
The runner also caps each phoneme at 100 deterministically sampled occurrences,
which balances common and uncommon phonemes and keeps the high-dimensional
input-window control within Colab memory. Full eligible counts are retained in
the output table.

CTC timing comes from the decoder itself. It is a pragmatic alignment for
discovery, not independent phoneme timing. A positive hidden-state result must
be checked against the input control and, ultimately, raw-bin activity.

Run from the repository root:

```bash
python -m experiments.manifolds.run_export_analysis \
  "/path/to/willett_manifolds/export_name/gru_released" \
  "/path/to/output/trajectory_hidden" \
  --representation hidden
```

Repeat with `--representation input_windows` when the export was created with
`SAVE_INPUT_WINDOWS=True`.

## Interpretation ladder

A useful sequence of claims is:

1. **Decodable:** phonemes separate in the decoder state.
2. **Repeatable:** held-out repetitions follow similar paths after alignment.
3. **Neural:** the effect exists in raw 20 ms population activity, not only in
   decoder-created features.
4. **Dynamical:** path geometry predicts temporal evolution beyond static
   phoneme identity, session, duration, and neighboring phonemes.

Start by stratifying within subject and session. Then test leave-session-out
generalization and context-conditioned labels (central phoneme plus preceding
and following manner/phoneme). Speech coarticulation makes isolated phoneme
averages a deliberately coarse first pass.

## Next experiment if the first pass is positive

Add a raw-bin exporter after the exact evaluation preprocessing but before
temporal patching. Fit PCA/FA only on training sessions, project held-out
sessions, align trials with CTC centers, and compare against shuffled labels,
time-reversed paths, and matched-duration random centers. Only after those
controls is it worth trying GPFA, jPCA, LFADS-style models, or nonlinear
embeddings.

## Final cross-session robustness workflow

Use `notebooks/cross_session_trajectory_robustness.ipynb` for the confirmatory follow-up. It
reconstructs the covered 20 ms bins and uses five same-utterance null paths per
event, each constrained not to overlap the real path. Leave-one-session-out
folds are the inferential units, with shape-only separation plus balanced
phoneme/category classification. The real path and every null draw use the same
sampled event pairs when estimating separation. Repeated 25% session splits are
retained as descriptive stability checks rather than independent statistical
observations.
