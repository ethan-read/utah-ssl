# Canonical Cache Specification

This document defines the current physical cache and manifest contract for the
Utah-array datasets. Canonical training views, clipped-FP16 SBP roots, and
normalization artifacts are indexed in
[`cache_and_stats_inventory.md`](cache_and_stats_inventory.md).

## Roots and availability

The canonical raw full-corpus cache is:

- local: `/Users/home/thesis/data/cache_v1`
- Colab Drive: `/content/drive/MyDrive/utah_ssl/data/cache_v1`

The canonical sigma-2 pre-smoothed full-corpus cache is available locally at:

```text
/Users/home/thesis/data/cache_v1_smoothed_sigma2p0
```

The Drive path below contains pre-smoothed Brain-to-Text source data, not a
complete smoothed copy of every auxiliary dataset:

```text
/content/drive/MyDrive/utah_ssl/data/cache_v1_smoothed_sigma2p0
```

BIT-style Modal jobs use a copy of the local full-corpus smoothed cache in the
documented cache volume. New Brain-to-Text SBP jobs use the versioned
clipped-FP16 roots listed in the inventory rather than FP32 SBP from these
source roots.

## Dataset directory contract

Each dataset directory contains:

```text
<cache_root>/<dataset>/
├── metadata.json
├── manifest.jsonl
└── shards/
```

- `metadata.json` records dataset-wide provenance, feature layout, and build
  choices.
- `manifest.jsonl` is the example-level index and points into the shards.
- `shards/` stores modality and auxiliary arrays.

Generic code must depend only on this core. Extra files or dataset-specific
manifest fields are optional and must not become global requirements.

## Global invariants

- Cached neural data uses 20 ms bins.
- Normalization is performed at load or training time using a matching reusable
  statistics artifact; it is not baked into the feature arrays.
- Modalities and auxiliary signals are stored as separate arrays rather than a
  single concatenated tensor.
- A manifest entry identifies the shard and example index required to retrieve
  one logical example.
- Shard boundaries are storage details and need not match source sessions.
- Feature width, available modalities, labels, and optional behavioral arrays
  vary by dataset.

## Current dataset inventory

| Dataset | Neural signals | Physical width | Examples | Sessions | Labels |
|---|---|---:|---:|---:|---|
| `brain2text24` | TX, SBP | 128 each | 16,088 | 28 | phonemes |
| `brain2text25` source | TX, SBP | 256 each | 10,948 | 45 | 9,498 labeled |
| `000950` | TX | 192 | 728 | 47 | none |
| `motor_data` | TX, SBP | 128 or 256 each | 16,420 | 21 | none |
| `plug_n_play` | TX | 192 | 1,315 | 31 | no phoneme labels |
| `unsupervised_cursor_recalibration_offline` | TX | 192 | 69,157 | 103 | none |
| `unsupervised_cursor_recalibration_online` | TX | 192 | 97 | 11 | none |
| `willett_handwriting` | TX | 192 | 5,412 | 10 | text |

Physical widths describe the source cache. Both active Brain-to-Text datasets
use 128-channel area-6v views, and active Brain-to-Text SBP comes from the
clipped-FP16 derivatives.

### Brain-to-Text 2024

- Source splits: `competition_train` 8,800; `competition_test` 880; no source
  split 6,408.
- Raw trial boundaries use `time_offsets.npy`.
- Phoneme labels use `phoneme_offsets.npy` and `phoneme_ids.npy`.
- TX and SBP are stored separately.

### Brain-to-Text 2025

- Source splits: `train` 8,072; `val` 1,426; `test` 1,450.
- The 256-channel source folder is not an active model input.
- Active jobs use the versioned 128-channel area-6v projection and the
  corresponding clipped-FP16 SBP derivative.

### Auxiliary datasets

- `000950` is already 20 ms binned, TX-only handwriting data and includes an
  `eval_mask`.
- `motor_data` has no single global feature width: shards contain either 128 or
  256 channels per available modality. Source files at 10 ms were converted to
  20 ms by summing TX and averaging SBP. A multi-dataset model must apply an
  explicit width-harmonization policy.
- `plug_n_play` is TX-only speech-related data whose manifest includes decoder
  outputs and decoder-condition metadata.
- `unsupervised_cursor_recalibration_offline` stores cursor and target
  positions, decoded velocity, target state, and geometry. Released rates were
  converted to approximate counts; empty spike-power data is not cached.
- `unsupervised_cursor_recalibration_online` stores cursor, target, velocity,
  decoder-state, clock, and target-radius arrays. Recorded clock gaps are
  preserved.
- `willett_handwriting` was converted from 10 ms to 20 ms. Its text encoding
  uses `>` for space and `~` for period.

## Manifest contract

The common manifest fields are:

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
- `has_tx`, `has_sbp`, `n_tx_features`, and `n_sbp_features` when applicable

Speech and trial datasets may also provide fields such as `transcript`,
`sentence_label`, `task_family`, `task_name`, `trial_num`, `trial_key`, and
`block_num`. Behavioral datasets add their own optional fields. Loaders must
treat these as dataset-specific rather than imposing one rigid manifest schema.

## Area and SBP policy

- Brain-to-Text 2024 and Brain-to-Text 2025 always use 128 area-6v channels.
- Brain-to-Text SBP is loaded from validated clipped-FP16 raw or pre-smoothed
  caches.
- Full-width Brain-to-Text source arrays are not an experimental alternative.
- Auxiliary feature widths and anatomical meanings remain dataset-dependent.
- Multi-dataset models must explicitly project or pad heterogeneous TX inputs
  according to their declared `SignalSpec`.

## Smoothing policy

- Raw and sigma-2 pre-smoothed caches are distinct data views.
- Runtime Gaussian smoothing must be zero when a pre-smoothed cache is used.
- Statistics must be computed from the exact raw or smoothed cache view used by
  the model.
- The local sigma-2 root supplies the full broad-TX corpus used for BIT-style
  preparation; the similarly named Drive source root is not a complete
  auxiliary-dataset mirror.

## Modeling implications

Broad Utah-array pretraining uses TX because several auxiliary datasets have no
SBP. Downstream Brain-to-Text decoding defaults to area-6v clipped-FP16 SBP;
TX-containing downstream inputs are explicit modality comparisons.

Code must not represent the broad corpus as a universal TX+SBP tensor with
silently missing SBP. The dataset plan and signal contract must enumerate the
included datasets, splits, widths, and any explicit padding or projection.
