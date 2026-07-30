# Cache And Stats Inventory

This note records the cache roots and normalization-stat artifacts that are
still relevant for current Utah SSL notebook work.

For the fuller Drive-backed dataset-by-dataset cache contract, see
`docs/notes/canonical_drive_cache_spec.md`.

## Area-6v-Only Feature Policy

Active Brain2Text24 cache roots should now be hard-migrated to area 6v only.
The Stanford `speechBCI` converter documents that the first `128` columns of
both `tx1` and `spikePow` are area 6v. The paper reports little decodable
information in area 44, so active SSL/POSSM/Willett runs should not spend
training time on the BA44/IFG columns.

Current feature-mode semantics are therefore:

- `tx_only`: first `128` area-6v TX features
- `sbp_only`: first `128` area-6v SBP features
- `tx_sbp`: first `128` area-6v TX features plus first `128` area-6v SBP
  features, for `256` total input features

Older full-array cache artifacts with `256` TX + `256` SBP columns should be
trimmed in place with
`analysis/active/ssl_experiments/ssl_core/scripts/trim_area6v_cache.py`. Recompute all reusable
normalization stats after trimming; checkpoints trained with the old
`256`-dimensional `tx_only` or `512`-dimensional `tx_sbp` layouts should not be
resumed.

## Canonical Raw Cache

### Local
- path: `/Users/home/thesis/data/cache_v1`
- status: valid canonical raw cache root
- verified for `brain2text24`: yes
- notes:
  - `brain2text24` has `16088` rows and `28` sessions
  - supports `segment_bins=80`
  - after area-6v migration, `tx_only`, `sbp_only`, and `tx_sbp` sampling
    should use `128` columns per requested modality at most

### Google Drive
- path: `/content/drive/MyDrive/utah_ssl/data/cache_v1`
- intended role: canonical raw cache used by Colab notebooks
- expected contents:
  - dataset folders such as `brain2text24/`
  - per-dataset `manifest.jsonl`, `metadata.json`, and `shards/`

## Pre-Smoothed Cache

### Local
- path: `/Users/home/thesis/data/cache_v1_smoothed_sigma2p0`
- status: valid local pre-smoothed cache root for `brain2text24`
- expected use:
  - select this cache root directly
  - keep `gaussian_smoothing_sigma_bins=0.0`
  - do not apply runtime smoothing on top of it

### Google Drive
- path: `/content/drive/MyDrive/utah_ssl/data/cache_v1_smoothed_sigma2p0`
- intended role: pre-smoothed Colab cache root
- required contents:
  - `brain2text24/metadata.json`
  - `brain2text24/manifest.jsonl`
  - `brain2text24/shards/`

## Versioned POSSM Brain2Text25 Cache

The pooled POSSM workflow has a separate, non-destructive cache preparation
path. Run
`analysis/reference/possm/notebooks/s15_possm_pooled_cache_preparation.ipynb`
in Colab to create:

- `utah_ssl/data/cache_v1_possm_b2t25_area6v_v1`
- `utah_ssl/data/cache_v1_possm_b2t25_area6v_sigma2p0_v1`

These roots include only Brain2Text25, retain 128 area-6v TX and 128 area-6v
SBP channels, and use approximately 65 MiB fused shards to match the median
canonical Brain2Text24 shard size. Retained TX, SBP, labels, and offsets
preserve their source dtypes and values exactly. The raw and smoothed
destinations are projected/repacked independently so smoothing is not
regenerated across new example boundaries. Brain2Text24 continues to come
directly from the canonical roots and is not copied or modified.

The matching pooled session stats live under:

- `utah_ssl/data/stats/session_feature_stats/possm_b2t24_canonical_b2t25_area6v_sigma2p0_v1/sbp_only/session/`

The preparation notebook performs logical example-level validation before a
partial cache is promoted to its final root.

Validation requires every destination array to preserve the corresponding
projected source dtype and values exactly. Brain2Text24 retains exactly the
baseline cache representation and shard topology.

## Precomputed Session Stats

Reusable normalization statistics should now live with the data artifacts under
`utah_ssl/data/stats`, not under experiment-output folders. The older
`outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats` location is
legacy and may be absent in a fresh Drive organization.

### Canonical Drive Layout

Stage-1 BIT-style masked reconstruction sampling should follow the paper's
pretraining choice:

- feature mode: `tx_only`
- use `SBP` later for downstream speech decoding when available
- keep `20 ms` bins and session-level z-scoring
- the exact broad dataset/split plan is `BIT_STAGE1_DATASET_SPLITS` in
  `ssl_core/bit_cache_contract.py`; it positively lists the seven datasets and
  every allowed source split
- pretraining is `TX`-only because some datasets lack `SBP`

- Drive root:
  - `/content/drive/MyDrive/utah_ssl/data/stats/session_feature_stats/`
- current canonical Stage-1 target path:
  - `/content/drive/MyDrive/utah_ssl/data/stats/session_feature_stats/smoothed_sigma2p0/tx_only/session/ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.pt`
  - `/content/drive/MyDrive/utah_ssl/data/stats/session_feature_stats/smoothed_sigma2p0/tx_only/session/ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.json`

Stage-2 phoneme fine-tuning uses train-split global stats computed on the
released Willett `competition_train` rows and then applied to both train and
validation examples:

- Drive root:
  - `/content/drive/MyDrive/utah_ssl/data/stats/split_feature_stats/`
- current pooled-SBP POSSM Stage-2 target path:
  - `/content/drive/MyDrive/utah_ssl/data/stats/split_feature_stats/raw/brain2text24/competition_train/sbp_only/global_v1.pt`
  - `/content/drive/MyDrive/utah_ssl/data/stats/split_feature_stats/raw/brain2text24/competition_train/sbp_only/global_v1.json`

The historical TX baseline uses the parallel `tx_only/global_v1.{pt,json}`
location. The artifact metadata must match the complete `SignalSpec`, not just
the directory name.

The `.pt` files contain tensors/arrays. The paired `.json` files contain
human-readable provenance such as cache variant, feature mode, split policy,
dataset list, and creation time.

### Legacy Output-Root Locations

### Unsmoothed Stats
- Drive path:
  - `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats/session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_stable.pt`
- intended use:
  - legacy raw-cache normalized runs
  - set `USE_SMOOTHED_CACHE = False`
  - set `SESSION_STATS_VARIANT = 'unsmoothed'`

### Smoothed Stats
- Drive path:
  - `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats/session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_smooth_sigma2p0_stable.pt`
- intended use:
  - legacy sigma-2 pre-smoothed-cache normalized runs
  - set `USE_SMOOTHED_CACHE = True`
  - set `SESSION_STATS_VARIANT = 'smoothed'`
- requirement:
  - use only with the matching `cache_v1_smoothed_sigma2p0` root

## Current Notebook Defaults

### `analysis/reference/possm/notebooks/s6_possm_maskedreconstruction.ipynb`
- current intended setup:
  - stage 1 uses `USE_SMOOTHED_CACHE = True`
  - stage 1 keeps `GAUSSIAN_SMOOTHING_SIGMA_BINS = 0.0`
  - stage 1 uses session-level featurewise z-scoring stats; prefer the
    `utah_ssl/data/stats/session_feature_stats/...` artifact once created
  - stage 2 uses the raw cache root via `STAGE2_CACHE_ROOT = DEFAULT_CACHE_ROOT_RAW`
  - stage 2 uses train-split global z-scoring stats from `competition_train`
  - stage 2 applies Willett-style online input smoothing after normalization and training-time noise/offset augmentation
  - `FEATURE_MODE = 'tx_only'`
  - `BOUNDARY_KEY_MODE = 'session'`
  - the notebook contains a temporary maintenance cell to populate
    `/content/drive/MyDrive/utah_ssl/data/stats`
- rationale:
  - stage-1 SSL smoothing is precomputed in the selected cache
  - stage-2 CTC smoothing is online so that future white-noise and constant-offset augmentations are smoothed in the same order as Willett's decoder
  - runtime smoothing remains removed from the generic masked SSL helpers
  - session-level boundary keys are needed for day/session-specific adaptation and diagnostics

## Practical Rules

- Keep reusable stats in `utah_ssl/data/stats`; keep run checkpoints/logs in
  `utah_ssl/outputs/ssl_experiments`.
- Pair raw cache with unsmoothed stats.
- Pair sigma-2 pre-smoothed cache with sigma-2 smoothed session stats.
- Do not mix Stage-1 session stats with Stage-2 split/global stats; they have
  different scopes.
- Keep `gaussian_smoothing_sigma_bins=0.0` whenever using a pre-smoothed cache.
- Before long Colab runs, verify that the selected cache root contains `brain2text24/manifest.jsonl`, `brain2text24/metadata.json`, and `brain2text24/shards/`.
- If normalized samples look shifted, recompute stats from the exact cache root being used rather than mixing stats across cache variants.
- If a Brain2Text24 cache reports `256` TX or `256` SBP feature columns, treat
  it as stale full-array data and run the area-6v migration before training.
