# Cache And Stats Inventory

This note records the important Utah SSL cache and session-stats variants that are easy to confuse during Colab and local notebook work.

## Canonical Raw Cache

### Local
- path: `/Users/home/thesis/data/cache_v1`
- status: valid canonical raw cache root
- verified for `brain2text24`: yes
- notes:
  - audit confirmed `brain2text24` has `16088` rows, `28` sessions
  - supports `segment_bins=80`
  - both `tx_only` and `tx_sbp` sampling checks passed

### Google Drive
- path: `/content/drive/MyDrive/utah_ssl/data/cache_v1`
- intended role: canonical raw cache used by Colab notebooks
- expected contents:
  - dataset folders such as `brain2text24/`
  - per-dataset `manifest.jsonl`, `metadata.json`, `shards/`

## Pre-Smoothed Cache

### Local smoothed cache for `brain2text24`
- path: `/Users/home/thesis/utah-ssl/data/cache_v1_smoothed_sigma2p0`
- status: valid local pre-smoothed cache root
- verified for `brain2text24`: yes
- expected use:
  - pre-smoothed cache selection with `gaussian_smoothing_sigma_bins=0.0`
  - no runtime smoothing

### Google Drive smoothed cache root currently referenced by `s6`
- path: `/content/drive/MyDrive/utah_ssl/data/cache_v1_smoothed_sigma2p0`
- intended status after repair: valid pre-smoothed cache root for `brain2text24`
- required contents:
  - `brain2text24/metadata.json`
  - `brain2text24/manifest.jsonl`
  - `brain2text24/shards/`
- historical issue:
  - this folder previously contained only `brain2text25`, and later had `brain2text24/shards/` without the top-level manifest/metadata due to Drive sync problems
- verification:
  - check the required files above before long Colab runs

### Separately uploaded `brain2text24` folder on Drive
- status: historical temporary upload location
- observed contents at the time:
  - `metadata.json`
  - `manifest.jsonl`
  - `shards/`
- caveat:
  - if a `brain2text24` folder is not nested under `cache_v1_smoothed_sigma2p0/brain2text24`, the notebook cache loader will not use it

## Precomputed Session Stats

### Unsmoothed stats
- Drive path:
  - `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats/session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_stable.pt`
- status: present on Drive
- intended use:
  - raw-cache normalized runs
  - set `USE_SMOOTHED_CACHE = False`
  - set `SESSION_STATS_VARIANT = 'unsmoothed'`

### Smoothed stats
- Drive path:
  - `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/contrastive/precomputed_ssl_session_stats/session_feature_stats_session_featurewise_v1_refds000950_cap126682_tx256_sbp256_smooth_sigma2p0_stable.pt`
- status: present on Drive
- intended use:
  - smoothed-cache normalized runs
  - only valid when the selected cache root is the matching smoothed `brain2text24` cache

### Other smoothed stats variants
- examples present on Drive:
  - `..._smooth_sigma1p0_stride2_stable.pt`
  - `..._smooth_sigma1p5_stride2_stable.pt`
  - `..._smooth_sigma2p5_stride2_stable.pt`
- note:
  - these should only be paired with matching pre-smoothed cache roots

## Historical Notebook Behavior

### `s5_maskedreconstruction.ipynb`
- important detail:
  - the notebook trained from raw `cache_v1`
  - it did **not** switch the actual data root to `cache_v1_smoothed_sigma2p0`
- implication:
  - older `brain2text24` runs may have relied on historical runtime smoothing behavior or on a partially migrated raw-cache plus smoothed-stats setup

### `s6_possm_maskedreconstruction.ipynb`
- intended modern behavior:
  - use a selected cache root directly
  - keep `gaussian_smoothing_sigma_bins=0.0`
  - pair raw cache with unsmoothed stats, or smoothed cache with matching smoothed stats
- temporary fallback:
  - if the smoothed Drive cache is broken, a raw-cache smoke test can run with:
    - `USE_SMOOTHED_CACHE = False`
    - `SESSION_STATS_VARIANT = 'unsmoothed'` if the unsmoothed stats load cleanly
    - or `SESSION_STATS_VARIANT = 'none'` if forced to recompute stats in memory

## Practical Rules
- Do not assume a folder named `cache_v1_smoothed_sigma2p0` is valid for `brain2text24`; verify `manifest.jsonl`, `metadata.json`, and `shards/` exist under `brain2text24/`.
- For normalized raw-cache runs, prefer the unsmoothed precomputed stats file over recomputation when it matches the current feature mode.
- For normalized smoothed-cache runs, require both:
  - the correct smoothed `brain2text24` cache root
  - the matching smoothed stats artifact
- When in doubt, run `analysis/active/ssl_experiments/audit_cache_roots.py` before long Colab jobs.
