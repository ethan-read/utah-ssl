# Cross-Trained Mamba Experiment Spec

## Goal

Build a supervised cross-session / cross-subject / cross-dataset phoneme
decoder for Brain-to-Text speech data that does the core architectural job of
the Tether Evo GRU model, but replaces the hierarchical GRU stack with a causal
Mamba sequence backbone.

The primary question is:

> Can a Mamba decoder, trained with Tether-Evo-style session/day affine
> alignment and hierarchical CTC supervision, match or improve the transfer
> behavior of the Tether Evo hierarchical GRU while staying closer to a compact
> streaming SSM-style decoder?

This is not a self-supervised pretraining experiment at first. The first pass is
a supervised model-design experiment that combines:

- the Tether Evo transfer recipe: learned per-day / per-subject affine input
  transforms, hierarchical phoneme feedback, and hierarchical CTC;
- the existing repo's Willett-style supervised CTC pipeline, splits,
  normalization, augmentation, and PER reporting;
- the existing Modal Mamba runtime work from `future_prediction_ssl`, especially
  fast-kernel verification before launching expensive jobs.

The first benchmark target is Brain2Text24 because the repo's current
supervised baselines and cleanest area-6v cache policy live there. The intended
scope is larger: cross-train on Brain2Text24, Brain2Text25, and the
Card/Willett speechBCI-style data where compatible labels and area-6v cache rows
are available.

## Source Threads

### Tether Evo Architecture

Reference note:

- `docs/paper_notes/tether_evo_b2t_architecture_notes.md`

Reusable ideas:

- Input features are native `20 ms` `TX + SBP` bins.
- A learned affine transform indexed by subject/day aligns neural activity into
  a shared space before the shared decoder.
- The shared temporal model is hierarchical, with intermediate phoneme
  classifiers.
- Intermediate phoneme probabilities are projected back into hidden space and
  fed into deeper blocks.
- Training uses CTC at multiple depths:
  - final CTC loss;
  - early auxiliary CTC loss;
  - middle auxiliary CTC loss weighted by `lambda = 0.3`.
- Adaptation regimes should separate:
  - train only new affine transforms;
  - fine-tune the full model;
  - train from scratch on the target participant/session.

Important caveat:

- Tether Evo's reported GRU stack uses bidirectional GRUs in the first two
  blocks, so it is not strictly causal despite some high-level causal language.
  Our first Mamba model should be causal by default, with an optional
  non-causal comparison only if the codebase later supports one cleanly.

### Existing Mamba / Modal Work

Reference files:

- `analysis/active/ssl_experiments/future_prediction_ssl/AGENTS.md`
- `analysis/active/ssl_experiments/future_prediction_ssl/design_decisions.md`
- `analysis/active/ssl_experiments/future_prediction_ssl/model.py`
- `analysis/active/ssl_experiments/ssm_ssl/model.py`
- `scripts/modal/run_future_prediction_ssl.py`

Reusable ideas:

- Use the generic `ssm_ssl` Mamba wrapper as the implementation anchor:
  `GenericSSMEncoder` and `MambaSequenceBackbone`.
- Mamba should be run causal.
- Use Hugging Face `MambaModel` through `inputs_embeds`.
- Before a Modal training run, verify optimized kernels in the actual remote
  image:
  - CUDA available;
  - `selective_state_update` present;
  - `selective_scan_fn` present;
  - `mamba_inner_fn` present;
  - `_lazy_load_causal_conv1d()` succeeds;
  - `MambaMixer` instantiation does not warn that it is using the slow path.
- Reuse the working candidate-package pattern:
  - CUDA image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`;
  - Python `3.10`;
  - `torch==2.8.0`;
  - `transformers==4.57.6` or the verified candidate;
  - `mamba-ssm==2.3.2.post1`;
  - `causal-conv1d==1.6.2.post1`;
  - install `causal-conv1d` and `mamba-ssm` with `--no-build-isolation`.
- Prefer the established Modal volume pattern:
  - cache volume: `utah-ssl-cache` mounted at `/vol/cache`;
  - output volume: `utah-ssl-outputs` mounted at `/vol/outputs`;
  - GPU request: `["L40S", "RTX-PRO-6000"]`.

### Existing Supervised Willett / S5 Recipe

Reference files:

- `docs/notes/willett_reconstruction_replication.md`
- `docs/notes/runpod_willett_s5_tx_sbp.md`
- `docs/notes/experiment_synthesis.md`
- `docs/notes/cache_and_stats_inventory.md`
- `docs/notes/canonical_drive_cache_spec.md`
- `analysis/active/ssl_experiments/willett_reconstruction/model.py`
- `analysis/active/ssl_experiments/willett_reconstruction/train.py`

Reusable ideas:

- Use canonical cache rows rather than Stanford TFRecords.
- Use `brain2text24`, `competition_train -> competition_test` for the first
  official single-dataset benchmark, with optional `competition_train_kfold`
  cross-validation.
- Treat cross-training as a first-class goal rather than a later rewrite:
  - `brain2text24` is the clean first evaluation anchor;
  - `brain2text25` can use the existing full-width cache if the loader selects
    only the area-6v feature columns at runtime;
  - Card/Willett speechBCI-style data should be included where the cache exposes
    compatible area-6v neural features and phoneme/text labels.
- Use area-6v `tx_sbp` as the main feature mode:
  - `128` TX channels;
  - `128` SBP channels;
  - total input width `256`.
- Do not assume all datasets already share this width:
  - `brain2text24` is currently area-6v-only: `128` TX + `128` SBP;
  - `brain2text25` is currently full-width in the canonical Drive cache:
    `256` TX + `256` SBP;
  - other Card/Willett-style sources may have their own channel layout and may
    need an explicit area-6v column-selection policy.
- Use train-derived normalization stats for validation; do not recompute stats
  from validation rows.
- Keep the existing input augmentation policy for supervised decoding:
  - Gaussian smoothing after normalization/augmentation as already implemented
    in the Willett pipeline;
  - white noise augmentation;
  - small per-channel constant offset augmentation.
- Use best validation PER as the main checkpoint-selection metric.
- Preserve the current per-adapter-key training sampler pattern: each optimizer
  step samples one adapter key and accumulates microbatches from that key. This
  is closer to the Stanford day-wise training regime than mixed-session batches.

Current strongest supervised baseline to beat or match:

- `willett_s5_tx_sbp_seed7_60k`;
- best visible PER around `0.25591` at step `56000`;
- final PER around `0.25736` at step `60000`.

## First-Pass Experiment Contract

### Dataset Scope

- cache root:
  - local: `/Users/home/thesis/data/cache_v1`
  - Modal: `/vol/cache/cache_v1`
- initial single-dataset benchmark:
  - dataset: `brain2text24`
  - split: `competition_train_test`
  - optional sensitivity run: `competition_train_kfold`, `5` folds
  - feature mode: `tx_sbp`
  - expected raw input dimension: `256`
- intended cross-training set:
  - `brain2text24`
  - `brain2text25` with runtime area-6v feature selection
  - Card/Willett speechBCI-style supervised data if represented in the cache or
    imported into the cache contract with labels and area-6v features
- boundary key:
  - start with `session` for within-dataset runs;
  - use a composite key for cross-training, such as
    `dataset_family:subject_id:session_date`;
  - evaluate `subject_if_available` only after confirming it does not merge days
    that should have separate affine transforms.

### Area-6v Feature Harmonization

Tether Evo's affine layer assumes every example enters the shared decoder in a
common feature space. For this project, that common space must be area 6v only.
Do not feed full-width `256 TX + 256 SBP` Brain2Text25 tensors into the model.
It is fine to read from the full-width cache, but the loader must slice the
feature columns down to area 6v before normalization, augmentation, adaptation,
or batching. Card/Willett-style imports need the same explicit area-6v channel
policy before admission.

Required policy:

```text
area-6v TX + area-6v SBP features [B, T, 256]
  -> session/day affine adapter
  -> shared hierarchical Mamba
```

Use `shared_dim=256` for all `tx_sbp` speech runs:

- first `128` area-6v TX channels;
- first `128` area-6v SBP channels;
- no BA44/IFG/full-array fallback.

For full-width cache arrays, "area-6v-only" means column/channel selection, not
example-row filtering:

```python
tx_area6v = tx[:, :128]
sbp_area6v = sbp[:, :128]
x = concatenate([tx_area6v, sbp_area6v], axis=-1)
```

Dataset admission rules:

1. `brain2text24` is already admitted because the active cache is area-6v-only.
2. `brain2text25` may be admitted from the current full-width canonical Drive
   cache if the loader has an explicit runtime selector that keeps only
   `tx[:, :128]` and `sbp[:, :128]`.
3. Card/Willett-style data is not admitted until its area-6v channel mapping and
   labels are represented in the cache contract.
4. Keep per-dataset normalization stats separate before the session/day affine
   transform; do not pool stats across datasets unless all have already been
   reduced to the same area-6v feature contract.

### Model

The first model should be a `TetherMambaPhonemeModel` or similarly named module
inside this folder, not a mutation of the existing `WillettPhonemeModel` until
the behavior is validated.

Suggested structure:

```text
x_t [B, T, 256]
  -> dataset/session/day area-6v affine adapter bank
  -> optional Softsign/dropout compatibility path
  -> Mamba block 1
  -> phoneme head 1
  -> softmax(l1) -> projection -> residual add
  -> Mamba block 2
  -> phoneme head 2
  -> softmax(l2) -> projection -> residual add
  -> Mamba block 3
  -> phoneme head 3
```

The adapter should have two modes:

- `affine`: a true Tether-style linear transform `W_(d,s) x + b_(d,s)`, with
  square identity initialization;
- `stanford_input_net`: the existing `Linear -> Softsign -> Dropout` adapter
  from `WillettPhonemeModel`.

Use `affine` as the paper-faithful first setting on Brain2Text24 and as the
default for multi-dataset training after every dataset has been reduced to
area-6v `tx_sbp`. Keep `stanford_input_net` as an ablation because it is
already proven in this repo's supervised code.

### Mamba Backbone

Start with three explicit Mamba stages instead of one monolithic stack. This
matches Tether Evo's hierarchical block semantics and makes auxiliary CTC heads
clean.

Initial suggested sizes:

- hidden size: `512` for the first serious run;
- state size: `64` or `128`;
- blocks: `3`;
- layers per block: `2`, `2`, `1` to mirror Tether Evo's GRU depth;
- dropout: `0.1`;
- direction: causal only;
- input projection: `LayerNorm -> Linear -> LayerNorm`, matching `GenericSSMEncoder`.

Do not start at Tether Evo's reported `d = 2048`. That would confound the first
Mamba test with a large capacity jump. Treat `1024` and `2048` as scale-up
runs after the 512-wide model is debugged.

### Temporal Interface

There are two plausible first-pass choices:

1. Native-bin mode, closest to Tether Evo:
   - feed one `20 ms` bin per Mamba token;
   - no Willett temporal patching;
   - CTC emits at native frame rate.
2. Willett patch mode, closest to the strongest local supervised pipeline:
   - apply session adapter before patching;
   - use `patch_size=14`, `patch_stride=4`;
   - feed flattened patches to Mamba.

Recommended sequence:

1. Implement native-bin mode first because this is the cleanest answer to
   "make a Mamba model that does what Tether Evo did with their GRU."
2. Add Willett patch mode as the first ablation because this repo's strongest
   supervised results depend heavily on the local temporal patch interface.

### Hierarchical CTC Objective

For logits `l1`, `l2`, `l3` and target phoneme sequence `y`:

```text
loss = CTC(l3, y) + 0.3 * CTC(l2, y) + CTC(l1, y)
```

Record each component separately:

- `train_ctc_l1_bpphone`
- `train_ctc_l2_bpphone`
- `train_ctc_l3_bpphone`
- `train_ctc_total_bpphone`
- validation equivalents where feasible
- PER from final logits `l3`
- optionally PER from `l1` and `l2` for diagnostics

Use final logits `l3` for checkpoint selection.

### Phoneme Feedback

For each auxiliary head:

```text
p_i = softmax(l_i)
feedback_i = Linear(p_i, hidden_size)
h_i = z_i + feedback_i
```

Open implementation choice:

- Use `softmax(l_i.detach())` for feedback in the first stability smoke if
  gradients through probabilities destabilize training.
- Use non-detached `softmax(l_i)` for the paper-faithful run.

Document which mode was used in every run name.

### Training Recipe

Start from the existing supervised Willett config, with Mamba-specific changes:

- optimizer: Adam;
- learning rate: start with `1e-3` for 512-wide Mamba, not Tether's `5e-3`;
- min learning rate: `1e-5` or `1e-4`;
- warmup: `1000` steps;
- schedule: cosine decay;
- weight decay: `1e-5`;
- Adam epsilon: keep the current Willett value available (`1e-1`) and compare
  against a normal deep-learning value (`1e-8`) if optimization looks odd;
- max grad norm: `10.0`;
- batch target: `64` examples per optimizer step via gradient accumulation;
- first smoke: `100-500` steps;
- first real run: `60000` steps;
- later Tether-scale run: `120000` steps only after the 60k run shows promise.

For cross-dataset runs, sampling should be balanced deliberately:

- start with dataset-balanced sampling so Brain2Text24 is not swamped by larger
  or easier datasets;
- within each selected dataset, sample one adapter key per optimizer step, as in
  the current Willett-style per-day loop;
- report per-dataset training loss and validation metrics whenever labels are
  available.

### Adaptation Regimes

The key scientific value is cross-training and adaptation, so the experiment
should not stop at one pooled supervised run.

Minimum regimes:

1. `pooled_full`: train all adapters and Mamba weights on all training sessions.
2. `new_session_affine_only`: freeze a trained shared Mamba; train only a new
   affine adapter for held-out sessions or held-out fold rows.
3. `new_session_full_finetune`: initialize from pooled model and fine-tune all
   parameters on the target session/fold.
4. `target_scratch`: train the same architecture from scratch on the target
   session/fold.
5. `cross_dataset_affine_only`: train shared Mamba on source datasets, then
   freeze it and train only affine adapters on a held-out area-6v dataset such
   as runtime-sliced Brain2Text25 or imported Card/Willett-style data.
6. `cross_dataset_full_finetune`: initialize from the cross-trained model and
   fine-tune all parameters on the held-out target dataset.

The main Tether-style claim lives in the gap between `new_session_affine_only`
and `target_scratch`: if an affine-only adapter does well, the shared Mamba has
learned a useful cross-trained speech representation.

The stronger cross-dataset claim lives in the gap between
`cross_dataset_affine_only` and `target_scratch`: if a frozen shared Mamba plus
a new area-6v dataset/session affine adapter is competitive, the model is doing
the transfer work Tether Evo was designed to test.

## Baselines

Required before making a claim:

- existing supervised S5 `tx_sbp` result (`PER ~= 0.256`);
- existing Willett GRU recipe if rerun under the same current cache and split;
- Tether-Mamba without hierarchical feedback, final CTC only;
- Tether-Mamba with shared adapter only, no per-session affine bank;
- native-bin Tether-Mamba vs Willett-patch Tether-Mamba.
- Brain2Text24-only Tether-Mamba vs Brain2Text24+Brain2Text25 cross-trained
  Tether-Mamba evaluated back on Brain2Text24.
- Brain2Text25 target-scratch vs Brain2Text24-to-Brain2Text25 affine-only
  adaptation, once Brain2Text25 runtime area-6v selection and labels/splits are
  wired into the trainer.

Useful later:

- Mamba initialized from future-prediction SSL weights, then trained with this
  hierarchical CTC objective;
- affine-only adaptation with Mamba vs affine-only adaptation with GRU;
- larger hidden sizes `1024` and `2048`.
- Card/Willett-style held-out adaptation if the data is imported into the same
  area-6v cache/label contract.

## Modal Run Contract

Create a dedicated launcher only after local unit tests pass:

- `scripts/modal/run_cross_trained_mamba.py`

The launcher should copy:

- `analysis/active/ssl_experiments/cross_trained_mamba`;
- `analysis/active/ssl_experiments/willett_reconstruction`;
- `analysis/active/ssl_experiments/ssl_core`;
- `analysis/active/ssl_experiments/ssm_ssl`;
- any stats recomputation helpers it imports.

Before training, the launcher must run the same Mamba fast-kernel diagnostic
used by `scripts/modal/run_future_prediction_ssl.py` and fail closed if the
fast path is unavailable.

Default Modal paths:

```text
cache root:  /vol/cache/cache_v1
stats root:  /vol/cache/stats
output root: /vol/outputs/ssl_experiments/modal_cross_trained_mamba
```

Default run names should encode:

- dataset set: `b2t24`, `b2t24b2t25`, or `b2t24b2t25card`;
- feature mode: `txsbp`;
- temporal interface: `native20ms` or `patch14s4`;
- model width: `h512`;
- hierarchy: `hctc`;
- feedback: `fb` or `fbdetach`;
- adapter mode: `affine` or `inputnet`;
- seed;
- steps.

Example:

```text
cross_mamba_b2t24_txsbp_native20ms_h512_hctc_fb_affine_seed7_60k
cross_mamba_b2t24b2t25_txsbp_area6v_native20ms_h512_hctc_fb_affine_seed7_60k
```

## Implementation Checklist

1. Add a package under this folder:
   - `config.py`
   - `model.py`
   - `train.py`
   - `tests/test_cross_trained_mamba.py`
2. Reuse the Willett data/problem builders instead of writing a new cache
   reader for Brain2Text24, then generalize the problem builder to multiple
   labeled datasets.
3. Add a dataset-aware manifest filter that can select `brain2text24`,
   runtime-sliced `brain2text25`, and imported area-6v Card/Willett-style
   labeled rows.
4. Add a feature-selection guard that rejects model inputs wider than
   `128 TX + 128 SBP`, while allowing full-width caches to be sliced before the
   model sees them.
5. Implement an affine adapter bank with deterministic hashed module names, like
   the existing session adapter bank.
6. Implement hierarchical Mamba stages and verify output shapes:
   - native-bin token lengths equal input lengths;
   - patch token lengths match `ssl_core.patching`;
   - all three logits tensors have shape `[B, T_tokens, vocab]`.
7. Implement hierarchical CTC loss and component logging.
8. Add smoke tests for:
   - adapter routing;
   - full-width Brain2Text25 cache arrays sliced to `128 TX + 128 SBP`;
   - unsliced full-width tensors rejected before model input;
   - no-session fallback;
   - feedback projection shape;
   - detach vs non-detach feedback;
   - checkpoint save/load;
   - a tiny train/eval loop.
9. Add a local CPU smoke command.
10. Add Modal launcher with fast-kernel verification.
11. Run a tiny Modal smoke.
12. Launch the first real Brain2Text24 `60k` run.
13. Launch a Brain2Text24+Brain2Text25 cross-training smoke after the
    Brain2Text25 runtime area-6v selector is tested.

## Success Criteria

Plumbing success:

- local tests pass;
- tiny CPU or MPS smoke completes;
- Modal kernel verification passes;
- tiny Modal smoke writes progress, summary, and checkpoints.

Scientific first-pass success:

- native-bin hierarchical Mamba trains without blank collapse;
- final-head validation PER beats a no-feedback Mamba ablation;
- adaptation-only runs are meaningfully better than target-scratch under the
  same target-data budget.
- multi-dataset training can include Brain2Text25 without full-width features,
  silent area assumptions, or normalization leakage.

Strong result:

- `60k` hierarchical Mamba approaches or beats the existing supervised S5
  `tx_sbp` PER around `0.256`;
- affine-only adaptation remains useful when the shared Mamba is frozen.
- Brain2Text24+Brain2Text25 cross-training improves held-out dataset adaptation
  relative to target-scratch or Brain2Text24-only initialization.

## Non-Goals For The First Pass

- No LLM decoder.
- No WFST/5-gram language-model integration in the first model-training loop.
- No self-supervised pretraining unless the supervised architecture is already
  training cleanly.
- No broad multi-dataset cache changes.
- No change to canonical cache paths or stats layout.
