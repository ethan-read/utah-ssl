# Spalding et al. shared latent speech alignment

## Source identity

- **Reported:** Z. Spalding, S. Duraivel, S. Rahimpour, et al., "Shared
  latent representations of speech production for cross-patient speech
  decoding," *Nature Communications* (2026).
- DOI: <https://doi.org/10.1038/s41467-026-75455-1>
- Published online: 16 July 2026; the available paper is an unedited
  article-in-press version accepted 1 July 2026.
- Local primary source:
  `/Users/home/Downloads/shared latent representations of speech ecog.pdf`
- Official released code:
  <https://github.com/coganlab/cross_patient_speech_decoding>
- Released code inspected at commit
  `a70f1ae711cf3627cf9357d2a72868a74f2d774f`.
- Paper license: CC BY-NC-ND 4.0. Released-code license: MIT, copyright
  coganlab (2023).

The local PDF has 46 pages. Its figure captions render, but the main figure
art is absent from the corresponding figure pages in this early-access file.
The notes below therefore rely on the paper text, methods, captions, and
released code rather than visual interpretation of the plotted panels.

## Central idea

**Reported:** The paper does not align electrode coordinates directly. It
learns a functional alignment in two stages:

1. Fit a separate PCA basis to each patient's neural activity, producing a
   patient-specific low-dimensional trajectory.
2. Use CCA to find linear projections of two patients' PCA trajectories that
   are maximally correlated when the same speech condition and time point are
   placed in corresponding rows.

The source patient's latent activity can then be mapped into the target
patient's PCA space. Source trials transformed this way are appended to the
target patient's training data, while held-out target trials stay in the
native target space.

Calling this only "PCA alignment" is incomplete: PCA supplies separate latent
spaces, while condition-matched CCA supplies the cross-patient alignment.
Because speech labels establish the row correspondence used by CCA, this is a
supervised functional alignment even though CCA itself is fit without a class
loss.

## Reported data and preprocessing

Evidence: Results, "Alignment of cross-patient latent dynamics"; Methods,
"Participants," "Speech task," "High-gamma extraction," and "Latent dynamics
extraction."

- Eight native-English-speaking neurosurgical participants with intact overt
  speech performed an intraoperative repetition task.
- The task contained 52 non-word conditions. Each was a three-phoneme CVC or
  VCV sequence drawn from nine phonemes, normally repeated in three blocks.
- Neural recordings used 128- or 256-channel high-density micro-ECoG over
  sensorimotor cortex.
- The analyzed signal was high-gamma envelope power (70-150 Hz), downsampled
  to 200 Hz and baseline-subtracted.
- Offline latent analyses used a one-second window from -500 to +500 ms around
  measured speech-response onset.
- PCA was fit separately per patient over the channel dimension. The reported
  rule retained the number of components explaining 90% of input variance.
- Most latent and decoding analyses used only channels with significant
  response-related high-gamma activity. The spatial-map reconstruction used
  all channels.

## Exact PCA-CCA construction

Evidence: Methods, "Alignment of latent dynamics," equations 8-12; released
`aligned_decoding/alignment/AlignCCA.py` at the commit above.

For patient `i`, let trial activity have shape `(trials, time, channels)`.
Patient-specific PCA produces `(trials, time, k_i)`.

For a source-target pair:

1. Average repetitions within each full three-phoneme sequence.
2. Retain only sequence conditions present in both patients.
3. Flatten condition and time together. The same row in both matrices now
   denotes the same sequence and time point:
   `L_i.shape = (shared_conditions * time, k_i)`.
4. Mean-center each latent dimension, compute QR decompositions, and take the
   SVD of `Q_target.T @ Q_source`.
5. Form patient-specific CCA maps `M_i = pinv(R_i) @ U_i`, truncated to the
   smaller matrix rank.
6. Map source latent samples into the target PCA space as
   `L_source @ M_source @ pinv(M_target)`.

**Released-code behavior:** `AlignCCA(type="class")` implements this
condition-averaged construction. The same repository also implements a
matched-trial alternative, multiview CCA, and joint PCA.

**Released-code behavior differing from the paper text:** the current
`aligned_decoding/scripts/aligned_decode_svm.py` uses a fixed 30 PCs and
5-fold stratified cross-validation, whereas the paper's Figure 4 Methods
describe PCA retaining 90% variance and 20-fold cross-validation. Any local
adaptation should cite the chosen behavior explicitly rather than treating
the paper and current repository as identical.

## Main reported results

Evidence: Figures 2-6 and their associated Results sections.

- CCA increased cross-patient articulatory clustering and representational
  similarity relative to independently ordered PCA components.
- In the offline nine-way phoneme analysis, mean balanced accuracy was 0.24
  for target-patient-only models, 0.19 for unaligned pooled data, and 0.31 for
  aligned pooled data. The aligned condition exceeded the patient-specific
  condition (`p = 0.01`, FDR-corrected paired test; `n = 8`).
- The largest gain occurred for the patient with the least data. Alignment
  still helped when only about seven target trials were used.
- In the simulated real-time CTC-RNN analysis, mean PER was 87.1% for
  patient-specific training, 82.5% for unaligned pooling, and 79.4% after CCA
  alignment (`n = 7`). The patient-specific RNN was statistically
  indistinguishable from the shuffled-label chance control, so these absolute
  real-time results mainly support a relative data-scarcity effect.
- Alignment benefits depended strongly on spatial sampling. Aligned decoding
  significantly exceeded patient-specific decoding only below 3 mm simulated
  pitch and at simulated coverage of at least about 8 by 17 mm. Contact-size
  effects were weaker after correction.

The authors also used Tensor Maximum Entropy surrogate data, shuffled labels,
unaligned pooling, representational similarity, spatial-map reconstruction,
and spatial subsampling as controls.

## Interpretation limits

- **Reported limitation:** PCA and CCA are linear and treat time points as
  independent samples during fitting; they do not model nonlinear or temporal
  dynamics explicitly.
- **Reported limitation:** the study used acute overt speech from people
  without speech impairment, not chronic attempted speech from people with
  ALS.
- **Reported limitation:** dense, broad micro-ECoG sampling appeared important;
  successful transfer to much smaller or discontinuous arrays is not
  established by this paper.
- **Inferred:** the positive findings establish useful predictive alignment,
  not proof that the recovered coordinates are a unique biological manifold.
- **Inferred:** label-shuffle controls do not make the alignment unsupervised.
  Full-sequence labels determine which condition-time rows are paired before
  CCA is fit.
- **Inferred:** the repeated stochastic t-SNE silhouette analysis is a
  visualization-oriented secondary analysis. Decoder performance on held-out
  target data is the more relevant endpoint.
- **Inferred:** a trial-level split can allow the same repeated non-word
  condition to occur in source training data and target test data. That is
  compatible with the paper's low-data deployment question, but it is not a
  test of transfer to unseen linguistic conditions.

## Utah-array relevance

The following are local observations or proposed adaptations, not claims from
Spalding et al.

### Repository observations

- Brain-to-Text 2024 is participant T12: 16,088 examples and 28 sessions.
- Brain-to-Text 2025 is participant T15: 10,948 examples and 45 sessions;
  9,498 train/validation examples have labels.
- The active repository contract uses the first 128 speech-focused channels
  and prefers clipped-FP16 SBP for both datasets. SBP is the closest local
  analogue to the paper's high-gamma-power feature, but intracortical SBP is
  not identical to micro-ECoG high gamma.
- Between Brain-to-Text 2024 `competition_train` and Brain-to-Text 2025
  `train` plus `val`, there are 2,881 exact shared sentence prompts. This makes
  supervised cross-subject correspondence possible. Restricting B2T25 to
  `train` leaves 2,451 exact shared prompts.
- The correspondence is weaker than in the paper: most T12 training sentences
  occur only once, the prompts are long and variable in duration, and the
  cache contains phoneme sequences but not independent per-phoneme timestamps.
- Utah arrays have excellent local electrode density but much smaller and
  discontinuous cortical coverage than the broad micro-ECoG grids emphasized
  by the paper. Alignment is plausible, but this is a meaningful modality and
  sampling shift rather than a near-replication.
- Multi-session drift is substantial enough that a subject-level PCA could
  spend variance on day effects. Session-matched normalization and explicit
  split control must precede any interpretation of a cross-subject manifold.

### Recommended first adaptation

**Adapted proposal:** start with a geometry/linear-probe pilot, not a full CTC
decoder run.

1. Use canonical area-6v clipped-FP16 SBP. Fit normalization, PCA, temporal
   alignment, and CCA on development-training data only.
2. Use B2T25 `train` as source data. Keep B2T25 `val` available for a
   reverse-direction check, and do not touch either unlabeled test split
   during method development.
3. Avoid pairing raw sentence bin `t` across participants. Derive fixed-length
   phoneme-event windows with a frozen, training-only CTC/Viterbi alignment,
   or use a predeclared temporal-warping method. Condition rows should encode
   at least phoneme identity and relative event time; neighboring phoneme
   context should be tested because coarticulation is strong.
4. Fit separate subject PCA bases, then pairwise source-to-target CCA. Begin
   with the paper-faithful 90%-variance rule, but compare a fixed, nested-
   selected component count and regularized CCA because day-related variance
   and small canonical correlations can make ordinary CCA unstable.
5. Compare target-only, naive pooled PCA, CCA-aligned pooling, joint PCA or
   Procrustes, and shuffled-correspondence controls with identical target
   training data and compute.
6. Evaluate on held-out target sessions and on sentence prompts excluded from
   both subjects' alignment data. The second split is essential to distinguish
   a reusable phonetic alignment from shared-prompt memorization.
7. Treat cross-subject RSA and canonical correlations as diagnostics. Require
   improvement in held-out target linear phoneme decoding or PER before
   claiming that the alignment is useful.

The existing `experiments/manifolds` event-alignment and trajectory controls
are relevant scaffolding, but any positive transfer result should ultimately
be owned by a dedicated cross-subject alignment experiment with its own split
and provenance documentation.

## Provenance and validation status

- **Reported:** paper methods and quantitative results above were checked
  against the local primary PDF.
- **Released-code behavior:** CCA, joint-PCA, and decoding behavior above were
  checked against the official repository at the exact commit listed above.
- **Repository observation:** dataset counts, sessions, subjects, feature
  contracts, and shared-sentence count were checked against the canonical
  local manifests and data documentation on 16 August 2026.
- **Unverified:** no Utah-array PCA-CCA alignment has been implemented or run.
  Feasibility remains a hypothesis.
- If official code is reused, preserve its MIT copyright and license notice.
  Do not copy figure assets or adapted paper text under the paper's
  CC BY-NC-ND license.
