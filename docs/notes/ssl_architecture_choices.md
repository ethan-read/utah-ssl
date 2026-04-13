# SSL Architecture Choices

This document records the architecture choices under consideration for the causal self-supervised decoder setup.

It is intentionally incremental. Each section should capture the current state of discussion without forcing decisions that have not been made yet.

## Scope

- target system: causal neural decoder
- training regime: self-supervised pretraining followed by downstream transfer / decoding experiments
- current inspirations:
  - [`BIT_architecture_notes.md`](../paper_notes/BIT_architecture_notes.md)
  - [`cortical_ssm_architecture_notes.md`](../paper_notes/cortical_ssm_architecture_notes.md)
  - [`POSSM_architecture_notes.md`](../paper_notes/POSSM_architecture_notes.md)

## Decision Log

### 1. Backbone Family

- status: decided
- decision: focus on state-space models rather than RNNs as the primary architecture family
- notes:
  - RNNs are not the main direction for this project
  - transformers are out of scope for now

### 2. Initial SSM Candidates

- status: decided
- decision: start with `S5` and `Mamba` as the first SSM backbones to test
- notes:
  - `S5` is motivated by the structure used in Cortical-SSM
  - `Mamba` is included as a second causal SSM-family baseline
  - the relative strengths of these models for speech decoding remain an open empirical question
  - for the specific self-supervised future-prediction objective, current evidence says to continue only with `Mamba`

### 3. Training Regime

- status: decided
- decision: use self-supervised training as the main pretraining regime
- notes:
  - this is motivated in part by BIT
  - the exact self-supervised objectives are not yet decided

### 4. Self-Supervised Objective

- status: partially decided
- decision: future prediction remains in scope only for `Mamba`; other SSL objectives remain open
- notes:
  - in current experiments, `S5` did not beat the trivial `predict 0` baseline on future prediction
  - future-prediction follow-up work should therefore be limited to `Mamba`
  - masked reconstruction, contrastive learning, and hybrid objectives remain open for later comparison

### 5. Input Modality

- status: decided
- decision: use both threshold crossings (`TX`) and spike band power (`SBP`) when available
- notes:
  - this follows the input choice used in BIT for downstream speech decoding
  - some candidate datasets do not provide `SBP`
  - the current decision is to include both modalities where available rather than restricting the project to `TX` only

### 6. Temporal Representation Unit

- status: decided
- decision: compare single-scale raw `20 ms` bins against single-scale causal temporal patches
- notes:
  - patching remains in scope as a first-class architectural question
  - the comparison should be between simple single-scale variants
  - multiscale or two-stream models that process both raw bins and patches simultaneously are deferred for now

### 7. TX / SBP Fusion

- status: decided
- decision: concatenate `TX` and `SBP` into a single per-timestep feature vector when both are available
- notes:
  - this follows the simplest reading of BIT's input handling
  - separate modality-specific branches are out of scope for the initial model family
  - datasets without `SBP` will still require a missing-modality handling policy later

### 8. Subject / Session Adaptation

- status: decided
- decision: include explicit learned input adaptation for both subject-level and session/day-level variability
- notes:
  - this is motivated by BIT's subject-specific boundary layers and the Tether Evo paper's subject/day-specific affine transforms
  - the starting point should be a lightweight affine adaptation mechanism rather than a large session-specific subnet
  - the shared backbone should not be forced to absorb all cross-subject and cross-day drift on its own

### 9. Form Of Input Adaptation

- status: decided
- decision: start with full learned affine input transforms as the baseline adaptation mechanism
- notes:
  - low-rank or factorized variants are deferred until after the full affine baseline is tested
  - the immediate goal is to establish whether explicit affine realignment helps in the causal self-supervised setting

### 10. Placement Of Input Adaptation

- status: decided
- decision: apply the affine adaptation on raw per-timestep neural features before any temporal patching
- notes:
  - the purpose of the transform is to align the neural feature space itself
  - patching, if used, should operate on already-aligned features rather than on session-misaligned inputs

### 11. Scope Of The Affine Transform

- status: decided
- decision: use one joint affine transform over the concatenated `TX + SBP` feature vector
- notes:
  - separate modality-specific affine transforms are out of scope for the initial model family
  - the joint transform can in principle learn cross-feature mixing between `TX` and `SBP`

### 12. Alignment Versus Projection

- status: decided
- decision: separate input alignment from model projection
- notes:
  - the affine transform should operate in the native concatenated feature space
  - after alignment, a shared learned projection should map features into the model width used by the SSM
  - the affine transform is for domain alignment, not for replacing the shared input projection layer

### 13. Shared Input Projection

- status: decided
- decision: use a shallow normalized linear projection after affine alignment
- notes:
  - the default input projection should be simple rather than a deep MLP
  - this choice is closest to the light projection style used in BIT and is cleaner than a heavy front-end stack
  - richer convolutional or multi-branch front ends are deferred

### 14. Additional Front-End Sequence Modeling

- status: decided
- decision: do not add an extra convolutional front-end module in the initial model family
- notes:
  - the projected sequence should feed directly into the `S5` or `Mamba` backbone
  - this keeps the first comparison focused on the SSM itself rather than on stacked temporal modules
  - causal convolutions remain a deferred option if the simple front end proves too weak

### 15. Initial Architecture Grid

- status: decided
- decision: evaluate both `S5` and `Mamba` with both raw-bin and causal-patch inputs
- notes:
  - the initial comparison grid is:
    - `S5` + raw `20 ms` bins
    - `S5` + causal patches
    - `Mamba` + raw `20 ms` bins
    - `Mamba` + causal patches
  - patching is an input-representation choice before the backbone, not a replacement for the backbone
  - for implementation simplicity, the raw-bin condition can be represented as `patch_size = 1`

### 16. Causal Patching Style

- status: decided
- decision: allow overlapping causal temporal patches
- notes:
  - patching remains causal and must not use future bins
  - non-overlapping patching is not required as the only option
  - overlapping patches should help preserve timing detail that might otherwise be lost

### 17. Patching Hyperparameters

- status: decided
- decision: treat both `patch_size` and `stride` as explicit architecture choices
- notes:
  - stride should not be fixed to `1` by default
  - the patching configuration should be specified as a `(patch_size, stride)` pair
  - both overlapping and non-overlapping causal patching remain available through this parameterization
  - for bookkeeping simplicity, the no-patching case should be represented as `patch_size = 1`
  - initial allowed patch sizes are `{1, 3, 5}`
  - initial allowed strides are `{1, 3, 5}`
  - valid combinations should satisfy `stride <= patch_size`

### 18. Normalization Strategy

- status: decided
- decision: separate preprocessing standardization from learned affine alignment, and keep architectural normalization minimal
- notes:
  - normalization and affine alignment are treated as different mechanisms with different roles
  - preprocessing standardization should stabilize feature scale and distribution
  - learned affine transforms should handle subject/session alignment rather than replace normalization entirely
  - the shared front end should avoid heavy normalization stacks before the backbone

### 19. Normalization Search Space

- status: decided
- decision: test only a small normalization set centered on standardization scope and post-projection normalization
- notes:
  - standardization scope to test:
    - subject-level featurewise z-score
    - session-level featurewise z-score
  - post-projection normalization to test:
    - none
    - `RMSNorm` after the shared projection
  - backbone-internal normalization should be left to the default design of `S5` and `Mamba`
  - pre-projection `LayerNorm` is out of scope for the initial model family
  - batch normalization is out of scope
  - Cortical-SSM-style temporal-dimension normalization is out of scope for the initial model family

### 20. Default Normalization Configuration

- status: decided
- decision: start from subject-level featurewise standardization plus post-projection `RMSNorm`
- notes:
  - default order:
    - preprocessing standardization
    - joint subject/session affine alignment
    - optional patching
    - shared linear projection
    - `RMSNorm`
    - `S5` or `Mamba`
  - this is meant to stabilize optimization without aggressively undoing the affine alignment layer

### 21. Indexing Of Affine Adaptation

- status: decided
- decision: start with one learned affine transform per session key
- notes:
  - the initial implementation should not split adaptation into separate subject and session components
  - each session key can implicitly carry subject identity along with session/day identity
  - factorized subject-plus-session variants are deferred until later

### 22. Backbone Output Interface

- status: decided
- decision: the backbone should emit a full causal hidden-state sequence rather than collapsing the entire trial to one final state
- notes:
  - this keeps the architecture compatible with multiple self-supervised objectives
  - timestep-level or patch-level hidden states should remain available to later heads
  - collapsing to a single final state inside the backbone would unnecessarily restrict later objective choices

### 23. Output Readout Head

- status: deferred
- decision: not yet chosen
- notes:
  - the exact readout used by the self-supervised objective should be decided after the objective family is chosen
  - possible later options include current-state prediction, recent-window readout, pooled segment readout, or objective-specific prediction heads

### 24. Backbone Comparison Budgeting

- status: decided
- decision: compare `S5` and `Mamba` under roughly similar model budgets rather than exact parameter matching
- notes:
  - the primary comparison should keep training conditions, data, and front-end structure aligned
  - exact parameter equality is not required
  - the two backbone families should stay within the same rough scale so that comparisons remain meaningful
  - exact parameter-matched comparisons can be added later as a secondary control if needed

### 25. Stage 1 S5 Reference Block

- status: decided
- decision: implement one canonical pure-PyTorch `S5` reference block before opening S5-internal architecture search
- notes:
  - the first `S5` integration should use an actual `S5` layer rather than a quieter `S4D` fallback
  - the first `S5` encoder should keep the current benchmark interface unchanged:
    - same projected token sequence input
    - same full causal hidden-state sequence output
    - same held-out-session phoneme benchmark
  - the reference `S5` block should start as:
    - pre-norm residual block
    - `S5` sequence module
    - small feedforward sublayer
    - standard skip / residual connection
  - the first implementation should stay pure PyTorch for portability across local, Colab, and rented GPU environments

### 26. Future S5-Specific Search

- status: decided
- decision: defer `S5`-internal architecture exploration until after the canonical reference block is stable in the benchmark
- notes:
  - future autoresearch can explore bounded `S5`-specific choices such as:
    - norm type
    - feedforward width
    - residual / skip style
    - other small block-level changes
  - these should not be opened as search axes during the very first `S5` integration pass
  - the purpose of the first `S5` pass is to establish that the backbone family itself works under the fixed benchmark contract

### 27. Non-Causal Variants

- status: deferred
- decision: postpone offline or otherwise non-causal pretraining/model variants until after the causal design matrix is established
- notes:
  - the first benchmark should focus on the causal architecture family and causal-compatible self-supervised objectives
  - non-causal designs remain interesting as future comparisons rather than immediate baseline targets
  - later non-causal experiments should be framed explicitly as follow-up ablations against the stabilized causal benchmark

### 28. Causal Self-Supervised Objective Shortlist

- status: decided
- decision: keep five causal-compatible objective families in scope for the initial SSL discussion
- notes:
  - multi-horizon future prediction in input space
  - contrastive predictive coding (`CPC` / `InfoNCE`) over future windows
  - latent future prediction
  - same-window augmentation-based contrastive learning
  - held-out electrode-group prediction

### 29. Held-Out Electrode-Group Prediction

- status: decided
- decision: include held-out electrode-group prediction as an explicit objective family, with masking applied to electrode feature groups rather than individual scalars
- notes:
  - this should be framed as held-out electrode-feature prediction rather than held-out neuron prediction for the binned `TX + SBP` setting
  - if an electrode has both `TX` and `SBP`, those features should be masked together as one group
  - masking should preferably span contiguous time windows rather than isolated single time points
  - this objective is especially attractive as an auxiliary loss rather than the only pretraining signal

### 30. Objective Combination Policy

- status: decided
- decision: do not combine all objectives at once; use a staged combination strategy
- notes:
  - the first pass should establish single-objective baselines before multi-loss hybrids
  - hybrid objectives should use one clear primary loss plus at most one auxiliary loss in the initial benchmark
  - later combinations can be explored only after the individual components are understood

### 31. Preferred Hybrid Combinations

- status: decided
- decision: prioritize a small number of structured hybrid objectives rather than an unrestricted loss soup
- notes:
  - preferred combination `A`:
    - primary loss: multi-horizon future prediction
    - auxiliary loss: held-out electrode-group prediction
  - preferred combination `B`:
    - primary loss: `CPC` / future `InfoNCE`
    - auxiliary loss: multi-horizon future prediction
  - preferred combination `C`:
    - primary loss: latent future prediction
    - optional auxiliary loss: held-out electrode-group prediction
  - same-window augmentation contrastive learning is lower priority and should not be mixed into the first hybrid benchmarks

### 32. Initial Objective Ranking

- status: decided
- decision: start with predictive objectives before moving to more complex contrastive or augmentation-heavy variants
- notes:
  - initial order of emphasis:
    - multi-horizon future prediction
    - `CPC` / future `InfoNCE`
    - held-out electrode-group prediction
    - latent future prediction
    - same-window augmentation contrastive learning
  - this ordering is meant to keep the first objective family aligned with the causal deployment goal and the current architecture simplicity constraints

### 33. SSL Head Design

- status: decided
- decision: use one shared front end and backbone, with lightweight objective-specific heads attached to the full causal hidden-state sequence
- notes:
  - objective heads should be small compared with the shared backbone
  - most heads should operate per timestep or per patch rather than on a single sequence summary
  - session-specific output heads are allowed when reconstructing native electrode-feature space

### 34. Objective Matrix

- status: decided
- decision: define each in-scope objective by a concrete head type and target space
- notes:
  - multi-horizon future prediction:
    - head: small per-step prediction MLP
    - target: future aligned-and-standardized input tokens at the same granularity as the model input representation
    - loss: modality-balanced regression loss averaged across horizons
  - `CPC` / future `InfoNCE`:
    - head: projection head for current states and projection head for future target states
    - target: future backbone states at chosen horizons
    - loss: horizon-averaged `InfoNCE` with in-batch negatives
  - held-out electrode-group prediction:
    - head: session-specific reconstruction head back to native electrode-feature space
    - target: masked electrode groups in aligned-and-standardized native feature space
    - loss: regression loss computed only on masked groups and masked time spans
  - latent future prediction:
    - head: predictor MLP from current state into latent target space
    - target: stop-gradient future latent targets from a target projection branch
    - loss: latent regression or cosine-distance loss averaged across horizons
  - same-window augmentation contrastive learning:
    - head: projection head on matched segment embeddings
    - target: second augmented view of the same segment
    - loss: contrastive loss over positive view pairs and in-batch negatives

### 35. Initial Targeting Conventions

- status: decided
- decision: use conventions that keep objectives comparable across patched and unpatched variants
- notes:
  - future targets should be defined at the same token granularity produced by the chosen input representation
  - horizons should be interpreted as short, medium, and longer-term future prediction targets rather than a single-step-only task
  - held-out reconstruction should mask electrode groups jointly across `TX` and `SBP` for the same electrode when both are present
  - losses in native feature space should balance `TX` and `SBP` contributions so that one modality does not dominate purely by scale or smoothness

### 36. Initial Loss-Weighting Rules

- status: decided
- decision: use simple weighting rules for the first benchmark and tune only after baseline behavior is understood
- notes:
  - single-objective runs:
    - total loss weight of the primary objective = `1.0`
  - multi-horizon losses:
    - average equally across horizons in the initial implementation
  - preferred hybrid combination `A`:
    - `L_total = 1.0 * L_future + 0.5 * L_holdout`
  - preferred hybrid combination `B`:
    - `L_total = 1.0 * L_cpc + 0.5 * L_future`
  - preferred hybrid combination `C`:
    - `L_total = 1.0 * L_latent + 0.25 * L_holdout`
  - the first pass should not introduce more than two active loss terms at once

### 37. First Objective Benchmark Plan

- status: decided
- decision: stage the first objective benchmark in three passes rather than searching all losses simultaneously
- notes:
  - pass `1`: single-objective baselines
    - multi-horizon future prediction
    - `CPC` / future `InfoNCE`
    - held-out electrode-group prediction
  - pass `2`: preferred hybrids
    - future prediction + held-out electrode-group prediction
    - `CPC` + future prediction
  - pass `3`: more complex objectives
    - latent future prediction
    - same-window augmentation contrastive learning
  - later objectives should only be promoted if the earlier simpler objective families are well understood

### 38. Future-Horizon Schedule

- status: decided
- decision: treat the future-horizon schedule as an explicit design axis for automated research
- notes:
  - this applies to future prediction, `CPC` / future `InfoNCE`, and latent future prediction
  - horizons should be defined in real time rather than raw token count so that patched and unpatched variants remain comparable
  - the search should be restricted to short and medium horizons in the initial benchmark
  - the search space should remain small and discrete rather than allowing arbitrary horizon values

### 39. Horizon Search Policy

- status: decided
- decision: keep horizon search constrained and subordinate to the main architecture matrix
- notes:
  - horizon schedules should not be a first-pass free axis in the main architecture matrix
  - the first pass should use one fixed short/medium horizon schedule for future-based objectives
  - only after a stable future-based objective family is identified should alternative short/medium schedules be explored
  - a small number of short-biased schedules is preferred over a large combinatorial sweep
  - horizon schedule is a lower-priority design axis than backbone family, patching choice, and objective family
  - exact schedule values can be finalized later in the automation configuration rather than fixed immediately in this document

## V1 Benchmark Scope

### Frozen For V1

- causal-only model family
- `S5` and `Mamba` as the only backbone families in scope
- concatenated `TX + SBP` input when both are available
- one joint affine transform per session key in native feature space
- shallow shared projection after alignment
- no additional convolutional front end
- backbone emits a full causal hidden-state sequence
- no non-causal pretraining variants
- no long-horizon future objectives
- no multiscale or two-stream models

### Primary Search Axes For V1

- backbone family:
  - `S5`
  - `Mamba`
- temporal representation:
  - `patch_size in {1, 3, 5}`
  - `stride in {1, 3, 5}`
  - valid only when `stride <= patch_size`
- objective family:
  - multi-horizon future prediction
  - `CPC` / future `InfoNCE`
  - held-out electrode-group prediction

### Secondary Search Axes For V1

- preprocessing standardization scope:
  - subject-level featurewise z-score
  - session-level featurewise z-score
- post-projection normalization:
  - none
  - `RMSNorm`
- hybrid objective usage:
  - single-objective
  - preferred hybrid `A`
  - preferred hybrid `B`

### Deferred Beyond V1

- offline or bidirectional pretraining variants
- long-horizon future objectives
- latent future prediction as a main benchmark axis
- augmentation-heavy contrastive learning as a main benchmark axis
- low-rank or factorized affine adaptation
- separate modality-specific processing branches
- heavy structured front ends inspired by EEG/ECoG models
