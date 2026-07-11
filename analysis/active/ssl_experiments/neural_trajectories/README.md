# Neural Trajectories in Attempted Speech

This folder tests whether phoneme-related population paths are repeatable. It
starts from the durable exports made by `s11_willett_representation_manifolds`
and keeps full per-utterance time order, which the pooled PCA plots discard.

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
PYTHONPATH=analysis/active/ssl_experiments \
python -m neural_trajectories.run_export_analysis \
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
