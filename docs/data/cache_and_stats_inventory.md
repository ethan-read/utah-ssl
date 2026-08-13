# Canonical Cache and Statistics Inventory

This file lists only the data and normalization artifacts that are canonical
for current work. Experiment-specific defaults belong in their experiment
folders. Historical paths belong in archived experiment notes, not here.

For the cache schema and dataset-level storage contract, see
[`canonical_drive_cache_spec.md`](canonical_drive_cache_spec.md). For the
software-level signal and dataset rules, see
[`signal_and_dataset_contracts.md`](signal_and_dataset_contracts.md).

## Acquisition status

All datasets currently in scope are publicly accessible and associated with
their source papers. A repository-local acquisition guide is deferred. When it
is added, use the dataset citations and links collected in the BIT paper as the
starting index and verify each source before documenting download commands.

## Signal and storage policy

Brain-to-Text 2024 and Brain-to-Text 2025 always use the first `128` area-6v
channels. The remaining BA44/IFG channels are outside the active data contract.

SBP is the default Brain-to-Text signal because it has been more reliable than
TX in the current experiments. Store Brain-to-Text SBP using the validated
clipped-FP16 representation; this reduces storage and I/O without a measurable
decoding-accuracy loss. Training may cast the stored arrays to its compute
dtype after loading.

Most auxiliary Utah-array datasets provide TX but not SBP. Broad multi-dataset
pretraining therefore uses TX. `tx_only` and `tx_sbp` remain available only for
explicit modality comparisons; they are not implicit Brain-to-Text defaults.

Current Brain-to-Text feature meanings are:

- `sbp_only`: 128 area-6v SBP features;
- `tx_only`: 128 area-6v TX features; and
- `tx_sbp`: 128 area-6v TX plus 128 area-6v SBP features.

## General cache roots

The canonical full-corpus raw cache roots are:

- local: `/Users/home/thesis/data/cache_v1`
- Colab Drive: `/content/drive/MyDrive/utah_ssl/data/cache_v1`

The canonical sigma-2 pre-smoothed full-corpus source/TX cache is:

- local: `/Users/home/thesis/data/cache_v1_smoothed_sigma2p0`

The similarly named Colab Drive path
`/content/drive/MyDrive/utah_ssl/data/cache_v1_smoothed_sigma2p0` contains
pre-smoothed Brain-to-Text source data but is not a complete copy of the
smoothed auxiliary corpus. BIT-style Modal jobs use a copy of the local
full-corpus smoothed cache in their cache volume.

The raw root is the source and broad-TX cache. When a pre-smoothed cache is
selected, set runtime smoothing to zero so it is not applied twice.

Each dataset directory must contain `manifest.jsonl`, `metadata.json`, and
`shards/`. The local full-corpus root above, rather than any repo-local sample,
is the canonical local inventory.

## Brain-to-Text clipped-FP16 SBP roots

Use these Drive roots for all new Brain-to-Text SBP work:

| Dataset | View | Drive cache root |
|---|---|---|
| Brain-to-Text 2024 | raw | `/content/drive/MyDrive/utah_ssl/data/cache_v1_sbpclip12500_fp16_raw` |
| Brain-to-Text 2024 | sigma-2 pre-smoothed | `/content/drive/MyDrive/utah_ssl/data/cache_v1_sbpclip12500_fp16_smoothed` |
| Brain-to-Text 2025 | raw | `/content/drive/MyDrive/utah_ssl/data/cache_v1_possm_b2t25_area6v_sbpclip12500_fp16_raw_v1` |
| Brain-to-Text 2025 | sigma-2 pre-smoothed | `/content/drive/MyDrive/utah_ssl/data/cache_v1_possm_b2t25_area6v_sbpclip12500_fp16_smoothed_v1` |

The `possm` substring in the Brain-to-Text 2025 directory names is historical;
these are model-independent reusable data artifacts. For a mixed Brain-to-Text
2024/2025 job, map each dataset to its listed canonical root rather than
assuming both datasets are valid under one parent cache.

## Canonical normalization statistics

Reusable statistics live beneath:

```text
/content/drive/MyDrive/utah_ssl/data/stats/
```

The `.pt` payload and `.json` provenance sidecar form one logical artifact and
must remain together. Artifact metadata must match the full `DatasetPlan`,
`SignalSpec`, cache identity, source splits, and normalization scope.

### Broad-TX Stage 1

BIT-style broad pretraining uses the positively enumerated
`BIT_STAGE1_DATASET_SPLITS` plan in `utah_ssl/bit_cache_contract.py`, TX only,
20 ms bins, sigma-2 pre-smoothed data, and session normalization.

Canonical artifact pair:

```text
data/stats/session_feature_stats/smoothed_sigma2p0/tx_only/session/
  ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.pt
  ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.json
```

### Pooled Brain-to-Text 2024/2025 SBP Stage 1

The current pooled recipe uses sigma-2 pre-smoothed clipped-FP16 SBP with
session normalization over Brain-to-Text 2024 `competition_train` and
Brain-to-Text 2025 `train` and `val`.

Canonical artifact pair:

```text
data/stats/session_feature_stats/
  sbpclip12500_fp16_smoothed_mixed_ec756023c1a3/sbp_only/session/
    ssl_pretrain_brain2text24_brain2text25_plan_91b09aa3e7_v2.pt
    ssl_pretrain_brain2text24_brain2text25_plan_91b09aa3e7_v2.json
```

### Brain-to-Text 2024 SBP Stage 2

Downstream phoneme fine-tuning uses raw clipped-FP16 SBP and global statistics
computed only from Brain-to-Text 2024 `competition_train`, then applies those
statistics to both training and validation examples.

Canonical artifact pair:

```text
data/stats/split_feature_stats/sbpclip12500_fp16_raw/
  brain2text24/competition_train/sbp_only/
    global_v1.pt
    global_v1.json
```

## Maintenance

Generate or validate statistics with:

```bash
python utah_ssl/scripts/recompute_feature_stats.py --help
```

Reusable computation, canonical paths, and artifact-loading APIs live in
`utah_ssl.stats`; the script above is the sole command-line entry point.
Stable cache variants and source signatures live in `utah_ssl.cache_identity`
because both cache copying and statistics validation consume them. Physical
cache access in `utah_ssl.cache` does not define normalization artifacts.
Prepared cache contexts are consumed by `utah_ssl.sampling` for segment
selection, normalization, and batching. Model-independent smoothing used to
construct physical cache views lives in `utah_ssl.signal_processing`; the
Willett decoder's distinct adapted smoothing recipe remains in
`utah_ssl.decoding_preprocessing`.

Use `--scope session` for pretraining normalization and `--scope global` for
train-split downstream normalization. Generate only artifacts required by an
actual pipeline; do not materialize every combination of cache, signal, scope,
and split.

Before a long run:

- verify the selected dataset directories contain their manifest, metadata,
  and shards;
- verify Brain-to-Text inputs expose exactly 128 channels per requested
  modality;
- verify SBP arrays use the canonical clipped-FP16 representation;
- pair raw caches with raw statistics and pre-smoothed caches with matching
  pre-smoothed statistics; and
- reject and regenerate any statistics whose metadata does not match the exact
  cache, signal, dataset plan, or split policy.

Run outputs are separate from reusable data artifacts. New completed-run
evidence belongs under the Drive layout documented in
[`../run_artifact_layout.md`](../run_artifact_layout.md).
