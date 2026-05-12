# Cache And Stats Inventory

This note records the cache roots and session-stats artifacts that are still relevant for current Utah SSL notebook work.

## Canonical Raw Cache

### Local
- path: `/Users/home/thesis/data/cache_v1`
- status: valid canonical raw cache root
- verified for `brain2text24`: yes
- notes:
  - `brain2text24` has `16088` rows and `28` sessions
  - supports `segment_bins=80`
  - both `tx_only` and `tx_sbp` sampling checks passed

### Google Drive
- path: `/content/drive/MyDrive/utah_ssl/data/cache_v1`
- intended role: canonical raw cache used by Colab notebooks
- expected contents:
  - dataset folders such as `brain2text24/`
  - per-dataset `manifest.jsonl`, `metadata.json`, and `shards/`

## Pre-Smoothed Cache

### Local
- path: `/Users/home/thesis/utah-ssl/data/cache_v1_smoothed_sigma2p0`
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

## Precomputed Session Stats

### Unsmoothed Stats
- Drive path:
  - `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats/session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_stable.pt`
- intended use:
  - raw-cache normalized runs
  - set `USE_SMOOTHED_CACHE = False`
  - set `SESSION_STATS_VARIANT = 'unsmoothed'`

### Smoothed Stats
- Drive path:
  - `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats/session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_smooth_sigma2p0_stable.pt`
- intended use:
  - sigma-2 pre-smoothed-cache normalized runs
  - set `USE_SMOOTHED_CACHE = True`
  - set `SESSION_STATS_VARIANT = 'smoothed'`
- requirement:
  - use only with the matching `cache_v1_smoothed_sigma2p0` root

## Current Notebook Defaults

### `s6_possm_maskedreconstruction.ipynb`
- current intended setup:
  - stage 1 uses `USE_SMOOTHED_CACHE = True`
  - stage 1 keeps `GAUSSIAN_SMOOTHING_SIGMA_BINS = 0.0`
  - stage 1 uses `SESSION_STATS_VARIANT = 'smoothed'`
  - stage 2 uses the raw cache root via `STAGE2_CACHE_ROOT = DEFAULT_CACHE_ROOT_RAW`
  - stage 2 applies Willett-style online input smoothing after normalization and training-time noise/offset augmentation
  - `FEATURE_MODE = 'tx_only'`
  - `BOUNDARY_KEY_MODE = 'session'`
- rationale:
  - stage-1 SSL smoothing is precomputed in the selected cache
  - stage-2 CTC smoothing is online so that future white-noise and constant-offset augmentations are smoothed in the same order as Willett's decoder
  - runtime smoothing remains removed from the generic masked SSL helpers
  - session-level boundary keys are needed for day/session-specific adaptation and diagnostics

## Practical Rules

- Pair raw cache with unsmoothed stats.
- Pair sigma-2 pre-smoothed cache with sigma-2 smoothed stats.
- Keep `gaussian_smoothing_sigma_bins=0.0` whenever using a pre-smoothed cache.
- Before long Colab runs, verify that the selected cache root contains `brain2text24/manifest.jsonl`, `brain2text24/metadata.json`, and `brain2text24/shards/`.
- If normalized samples look shifted, recompute stats from the exact cache root being used rather than mixing stats across cache variants.
