---
name: audit-decoding-data
description: Audit Utah-array decoding data contracts, cache roots, dataset mixtures and splits, neural signals and channels, smoothing, normalization statistics, and checkpoint compatibility. Use when adding or removing datasets, changing TX/SBP or channel selection, changing cache or stats paths, preparing joint Brain-to-Text 2024/2025 or transfer-learning data, reusing an unfamiliar or legacy cache, investigating missing-channel or stale-stat errors, or checking for split leakage before training. Do not use for routine unchanged launches or general model-result interpretation.
---

# Audit Decoding Data

Determine whether an experiment will consume exactly the intended examples and neural features with compatible preprocessing and normalization. Keep the audit read-only unless the user explicitly asks to regenerate or migrate artifacts.

## Define the contract before inspecting files

1. Work from the repository root and read `AGENTS.md`.
2. Read:
   - `docs/data/cache_and_stats_inventory.md`
   - `docs/data/signal_and_dataset_contracts.md`
   - `docs/data/canonical_drive_cache_spec.md`
   - the owning experiment's recipe, notebook configuration, launcher, and design notes
3. Write down the intended:
   - `DatasetPlan`, including every dataset and allowed source split;
   - `SignalSpec`, including mode, dimensions, column start, and missing-channel policy;
   - cache root assigned to each dataset;
   - bin width, segment length, smoothing state, and augmentation;
   - normalization scope, statistics path, fitting datasets/splits, and application splits;
   - supervised train, model-selection, and held-out partitions;
   - Stage-1-to-Stage-2 or checkpoint signal handoff.
4. Treat an absent or implicit choice as an audit finding. Do not infer a broad “all except test” dataset selection when a positive `DatasetPlan` can enumerate the intended sources.

For Brain-to-Text 2024 and 2025, require the first 128 area-6v channels and the canonical clipped-FP16 SBP roots for speech-focused work. Treat TX or TX+SBP as an explicit comparison, not an interchangeable default. Broad auxiliary-dataset pretraining may use TX and explicit zero-padding where the named recipe allows it.

## Choose the audit depth

Use the narrowest sufficient level:

- **Contract review:** inspect configuration, paths, metadata, and split definitions without reading arrays. Use for proposed changes and notebook review.
- **Sampled structural audit:** inspect manifests, metadata, representative shards, sampling eligibility, and statistics. Use before trusting a new or changed cache.
- **Deep array audit:** scan every shard for shape consistency, all-zero channels, and nonfinite values. Use only when corruption, channel loss, or conversion errors are plausible because it can be expensive.
- **Runtime compatibility audit:** instantiate the exact repository contract and use canonical loaders. Use before a long run or when a stale-stat/checkpoint error is under investigation.

Routine runs with an unchanged, previously validated contract need only the ordinary checks in `$launch-decoding-run`.

## Audit caches with the maintained CLI

Use `utah_ssl/scripts/audit_cache_roots.py`; do not recreate its checks manually. Start with representative shards and only add `--deep-array-check` when justified.

```bash
python utah_ssl/scripts/audit_cache_roots.py \
  --cache-root <primary-cache-root> \
  --stats-path <session-stats.pt> \
  --dataset brain2text24 \
  --compare-dataset brain2text25 \
  --segment-bins <required-segment-bins> \
  --feature-mode sbp_only \
  --output-json /tmp/utah_ssl_cache_audit.json
```

Adjust datasets, feature mode, statistics paths, and cache roots to the declared contract. Pass repeated `--cache-root` arguments when comparing a canonical source with a candidate conversion. Write temporary reports outside the repository; generated audit JSON is not a tracked research result.

Interpret findings against the requested `SignalSpec`. A missing unused modality may be irrelevant for a single-modality cache, while a missing requested modality is blocking. Do not accept dimension-based guesses when metadata provides an explicit signal specification.

The CLI should establish:

- manifest, metadata, and shard presence;
- row, session, feature-width, and segment-eligibility summaries;
- sampled or deep array consistency;
- source and structural signatures across roots;
- available TX/SBP modes and sampling viability;
- statistics dimensions, session keys, smoothing compatibility, and root comparisons.

Run `python -m unittest utah_ssl.tests.test_cache_audit -q` after changing the audit implementation itself, not for an ordinary data audit.

## Exercise exact runtime validation

The CLI is a broad diagnostic. Use the runtime APIs for the exact experiment contract:

1. Construct `DatasetPlan` and `SignalSpec` from `utah_ssl.experiment_contract`.
2. Construct `CacheAccessConfig` and call `prepare_cache_context` from `utah_ssl.cache` with the exact per-dataset root mapping.
3. For Stage-1 session normalization, call `load_precomputed_session_feature_stats_into_cache_context` from `utah_ssl.stats`.
4. For downstream global normalization, call `load_precomputed_split_feature_stats` from `utah_ssl.stats` with the exact train split, evaluation split, boundary mode, and split policy.
5. Treat loader rejection as a valid blocking result. Use the recompute command printed by the loader only if the user asks to regenerate the artifact.

Do not bypass a source signature, sidecar, dataset plan, signal, normalization-scope, split-policy, or tensor-dimension mismatch. Keep each `.pt` statistics payload with its `.json` provenance sidecar.

## Check scientific leakage and preprocessing

Verify separately:

- SSL exposure: which unlabeled recordings appear in pretraining;
- supervised exposure: which labels influence optimization;
- normalization exposure: which examples fit means and standard deviations;
- selection exposure: which partition selects checkpoints or hyperparameters;
- final evaluation exposure: whether the reported partition remained untouched.

Using an evaluation-distribution recording without labels during SSL may be a documented transductive design rather than an implementation error. Label it explicitly and avoid calling the resulting evaluation fully inductive. Any use of evaluation labels, or fitting downstream normalization on evaluation examples, is blocking unless that use is the declared object of study.

Confirm that raw caches pair with raw statistics and pre-smoothed caches pair with matching pre-smoothed statistics. Runtime smoothing must remain disabled for pre-smoothed caches. Ensure augmentation is applied at the intended stage rather than baked in twice.

## Check checkpoints and handoffs

Inspect trusted checkpoint metadata for `dataset_plan` and `signal_spec`. Match them to the consuming stage before resume or transfer.

- Reject silent modality or channel changes.
- Reject legacy checkpoints missing required contracts unless the owning workflow explicitly supports a documented migration.
- Treat broad-TX-to-SBP transfer as an explicit signal handoff requiring implementation and documentation, not as an identical resume contract.
- Do not infer compatibility from matching tensor shapes alone.

## Report findings

Return:

1. the resolved dataset and signal contract;
2. cache and statistics identities examined;
3. pass, warning, or blocking status for structure, channels, splits, normalization, smoothing, and checkpoint handoff;
4. leakage or transductive-use classification;
5. exact evidence for every blocking finding;
6. the smallest corrective action, without applying it unless requested.

Do not edit experiment results or draw model-performance conclusions from a data audit.
