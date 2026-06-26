# Canonical Drive Cache Spec

This note records the actual Google Drive cache layout currently used by the
active Willett notebook work. It is meant to be the source of truth for future
cache prep, smoothing, repacking, and stat-generation scripts.

Last verified: 2026-06-14

## Canonical Roots

### Raw cache root

- notebook path:
  - `/content/drive/MyDrive/utah_ssl/data/cache_v1`
- Drive folder:
  - `https://drive.google.com/drive/folders/19Z5ZQZaKY47vM708rJ4YtC32ZTvPqKJY`
- role:
  - canonical raw cache root used by the Willett Colab notebook
- important note:
  - despite the name `cache_v1`, this root is already a fused/repacked cache,
    not an old per-session tiny-shard layout

### Pre-smoothed cache root

- notebook path:
  - `/content/drive/MyDrive/utah_ssl/data/cache_v1_smoothed_sigma2p0`
- Drive folder:
  - `https://drive.google.com/drive/folders/1cv2RpwvFqR968vFhsKeGjUO-o9OJxP5R`
- role:
  - sigma-2 pre-smoothed cache root for workflows that want smoothing baked
    into the cached features
- current contents:
  - `brain2text24`
  - `brain2text25`

## Root-Level Contents

The canonical raw root currently contains:

- `000950`
- `brain2text24`
- `brain2text25`
- `motor_data`
- `plug_n_play`
- `unsupervised_cursor_recalibration_offline`
- `unsupervised_cursor_recalibration_online`
- `willett_handwriting`
- `repack_summary.json`

Interpretation:

- these are the datasets that currently define the canonical Drive cache
- `repack_summary.json` documents the fused-shard repack pass
- if a future local cache disagrees with this root, treat the Drive root as
  authoritative unless we intentionally replace it

## Dataset Folder Contract

Every dataset folder in the canonical raw root should contain:

- `metadata.json`
- `manifest.jsonl`
- `shards/`

That is the minimum contract scripts should assume.

Observed exceptions:

- `brain2text24` also contains:
  - `metadata.json.pre_area6v_backup`
  - `manifest.jsonl.pre_area6v_backup`
- `unsupervised_cursor_recalibration_offline` also contains:
  - `README.md`

## Global Cache Invariants

These are the broad rules that appear to hold across the current Drive cache.

- cached time resolution is standardized to `20 ms`
- normalization is generally deferred to load time rather than baked into the
  cached feature arrays
- shards store arrays by modality or auxiliary signal, not one monolithic blob
- manifests are the example-level index and metadata contract
- metadata files describe dataset-wide provenance, feature layout, and build
  choices
- the raw Drive root is already repacked into larger shards, so scripts should
  not assume one shard per source session

## Repack Status

The root-level `repack_summary.json` shows that the canonical Drive root has
already been fused into larger shards.

| Dataset | Old shards | New shards | Examples |
| --- | ---: | ---: | ---: |
| `000950` | 47 | 4 | 728 |
| `brain2text24` | 44 | 26 | 16088 |
| `brain2text25` | 127 | 69 | 10948 |
| `motor_data` | 65 | 26 | 16420 |
| `plug_n_play` | 31 | 4 | 1315 |
| `unsupervised_cursor_recalibration_offline` | 103 | 7 | 69157 |
| `unsupervised_cursor_recalibration_online` | 97 | 1 | 97 |
| `willett_handwriting` | 19 | 2 | 5412 |

Operational rule:

- when we say "canonical Drive cache", we now mean this fused layout, not the
  pre-repack source cache

## Per-Dataset Spec

### `brain2text24`

- modalities:
  - `tx`, `sbp`
- feature layout:
  - `128` TX
  - `128` SBP
  - `256` total
- examples:
  - `16088`
- sessions:
  - `28`
- labeled:
  - yes, all examples
- source split counts:
  - `competition_train`: `8800`
  - `competition_test`: `880`
  - `none`: `6408`
- important policy:
  - this cache has already been hard-migrated to area `6v` only
- migration detail:
  - first `128` columns retained
  - old BA44 or IFG columns `[128, 256)` removed for each modality
- build behavior:
  - raw trial boundaries preserved via `time_offsets.npy`
  - labels stored via `phoneme_offsets.npy` and `phoneme_ids.npy`
  - TX and SBP stored separately

This is the cleanest and most intentionally maintained dataset in the current
cache.

### `brain2text25`

- modalities:
  - `tx`, `sbp`
- feature layout:
  - `256` TX
  - `256` SBP
  - `512` total
- examples:
  - `10948`
- sessions:
  - `45`
- labeled:
  - `9498` labeled examples
- source split counts:
  - `train`: `8072`
  - `val`: `1426`
  - `test`: `1450`
- important caveat:
  - unlike `brain2text24`, this cache has not been area-6v-trimmed in the
    current canonical Drive version

So `brain2text24` and `brain2text25` do not currently share the same feature
width or area-selection policy.

### `000950`

- modalities:
  - `tx` only
- feature layout:
  - `192` TX
- examples:
  - `728`
- sessions:
  - `47`
- labeled:
  - no
- target type:
  - `none`
- special cached content:
  - `eval_mask`
- source split counts:
  - `held-in-calib`: `21`
  - `held-in-minival`: `21`
  - `held-out-calib`: `5`

This is already 20 ms binned handwriting data and appears suitable as a
pretraining-only TX dataset.

### `motor_data`

- modalities:
  - `tx`, `sbp`
- examples:
  - `16420`
- sessions:
  - `21`
- subjects:
  - `4`
- labeled:
  - no
- important caveat:
  - this dataset does not have one globally consistent feature width

Observed shard behavior:

- some shards have `128` TX + `128` SBP
- other shards have `256` TX + `256` SBP

Build behavior:

- source files came in both `10 ms` and `20 ms`
- `10 ms` files were converted to `20 ms`
- TX was summed when downsampling
- SBP was averaged when downsampling
- unsupported source bin sizes were excluded

Operational rule:

- do not treat `motor_data` as a uniform-width dataset without an explicit
  harmonization step

### `plug_n_play`

- modalities:
  - `tx` only
- feature layout:
  - `192` TX
- examples:
  - `1315`
- sessions:
  - `31`
- labeled:
  - no phoneme labels
- source split counts:
  - `no_recalibration`: `10`
  - `recalibration`: `10`
  - `seed_model_training`: `11`
- special manifest fields:
  - decoder outputs from LM and RNN pipelines
  - decoder condition metadata

This is a TX-only speech-related auxiliary dataset with rich text-decoder
metadata in the manifest.

### `unsupervised_cursor_recalibration_offline`

- modalities:
  - `tx` only for neural features
- feature layout:
  - `192` TX
- examples:
  - `69157`
- sessions:
  - `103`
- labeled:
  - no
- special cached arrays:
  - `cursor_pos`
  - `target_pos`
  - `dec_vel`
  - `on_target`
  - `target_size`
  - `cursor_size`
- source subset counts:
  - `historical`: `83`
  - `new`: `20`

Build behavior:

- source was offline closed-loop cursor-control data
- released rates were converted back to approximate counts by dividing by `50`
- spike power was empty and therefore not cached

### `unsupervised_cursor_recalibration_online`

- modalities:
  - `tx` only for neural features
- feature layout:
  - `192` TX
- examples:
  - `97`
- sessions:
  - `11`
- labeled:
  - no
- special cached arrays:
  - `target_pos`
  - `cursor_pos`
  - `cursor_vel`
  - `bias`
  - `state`
  - `xpc_clock`
  - `target_radius`
- special timing note:
  - timing gaps from `xpc_clock` were preserved rather than reconstructed away

### `willett_handwriting`

- modalities:
  - `tx` only
- feature layout:
  - `192` TX
- examples:
  - `5412`
- sessions:
  - `10`
- labeled:
  - text labels present
- source task counts:
  - `sentences`: `9`
  - `single_letters`: `9`
  - `straight_lines`: `1`
- build behavior:
  - source data was `10 ms`
  - cache was standardized to `20 ms`
  - raw strings use `>` for space and `~` for period

## Manifest Contract

The exact set of fields varies by dataset, but the Drive manifests share a
clear common skeleton.

Common fields observed across manifests include:

- `example_id`
- `dataset_family`
- `subject_id`
- `session_id`
- `session_date`
- `source_split`
- `bin_size_ms`
- `source_bin_size_ms`
- `shard_id`
- `shard_relpath`
- `example_index`
- `n_time_bins`
- `target_type`
- `has_labels`
- modality or feature-width fields such as:
  - `has_tx`
  - `has_sbp`
  - `n_tx_features`
  - `n_sbp_features`

Common speech-style or trial-style fields include:

- `transcript`
- `sentence_label`
- `task_family`
- `task_name`
- `trial_num`
- `trial_key`
- `block_num`

Dataset-specific optional fields also exist and should not be treated as
required globally. Examples include:

- `eval_true_bins` in `000950`
- decoder outputs in `plug_n_play`
- cursor or target metadata in cursor-control datasets
- cue or source-group metadata in `brain2text24`

Operational rule:

- new generic cache code should assume a common core plus dataset-specific
  optional fields, not one rigid manifest schema for every dataset

## Area Policy

The current Drive cache is not globally harmonized with respect to brain-area
selection.

- `brain2text24` is already area-6v-only
- `brain2text25` is still full-width in the current canonical Drive copy
- `motor_data` appears mixed-width across shards
- the TX-only auxiliary datasets mostly sit at `192` channels and do not share
  the same area semantics as the Brain2Text caches

Operational consequence:

- feature dimensionality should be treated as dataset-dependent
- any multi-dataset pretraining plan must explicitly decide whether to:
  - keep heterogeneous widths and project them
  - trim or map datasets into a common area policy
  - exclude datasets that do not match the intended input contract

## Smoothing Status

Current Drive-backed smoothing status is:

- `brain2text24`:
  - raw cache exists
  - sigma-2 pre-smoothed cache exists
- `brain2text25`:
  - raw cache exists
  - sigma-2 pre-smoothed cache exists
- all other datasets:
  - raw cache exists
  - no pre-smoothed canonical Drive copy was observed in the current smoothed
    root

Operational consequence:

- if we want BIT-style or S5-style pretraining on several datasets with a
  matched smoothed-cache policy, we still need a systematic smoothing pass for
  the non-Brain2Text datasets

## Practical Rules For Future Work

- treat the raw Drive `cache_v1` root as the canonical cache unless we
  explicitly replace it
- do not assume all datasets share the same feature width
- do not assume all datasets carry SBP
- do not assume all datasets are labeled
- do not assume all datasets have been area-6v-trimmed
- generate normalization stats from the exact cache variant in use
- keep reusable stats under `utah_ssl/data/stats`
- keep experimental outputs under `utah_ssl/outputs`
- if a local cache copy disagrees with this note, verify against the Drive root
  before changing training scripts

## BIT Interpretation Note

The current Drive cache contents and the BIT paper should be kept conceptually
separate.

What the BIT paper implies for modeling:

- stage-1 masked-reconstruction pretraining should be treated as `TX`-only
- downstream speech decoding may use `TX + SBP` when `SBP` is available
- this is because the pretraining corpus mixes datasets with and without `SBP`

So a BIT-faithful stage-1 cache spec is not:

- "one shared `tx_sbp` pretraining tensor with masked missing `SBP`"

It is instead:

- one shared `tx_only` pretraining cache contract across the heterogeneous
  corpus
- separate downstream speech-decoding configs that can use `tx_sbp` on
  Brain2Text-style datasets

That distinction matters because the paper explicitly says they fall back to
`TX`-only during pretraining when `SBP` is unavailable in part of the corpus.

## Most Important Takeaways

If we need a compact mental model:

1. the canonical Drive raw cache is already fused and is bigger than just
   `brain2text24`
2. `brain2text24` is the most curated dataset and is already area-6v-only
3. `brain2text25` is not yet aligned to the same area policy
4. only `brain2text24` and `brain2text25` currently have canonical smoothed
   Drive copies
5. `motor_data` is the main dataset that needs extra care because its cached
   feature width is not uniform across shards
