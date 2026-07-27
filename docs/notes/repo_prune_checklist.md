# Repo Prune Checklist

This checklist is for cleaning the repository tree while preserving the active
research direction. It assumes the current source of truth is:

- `docs/notes/experiment_synthesis.md`
- `docs/notes/cache_and_stats_inventory.md`
- `docs/notes/modal_bit_stage1_workflow.md`
- `analysis/active/ssl_experiments/AGENTS.md`

## Current Active Lines

Keep these in the active tree:

- `analysis/active/ssl_experiments/ssl_core`
- `analysis/active/ssl_experiments/ssm_ssl`
- `analysis/active/ssl_experiments/willett_reconstruction`
- `analysis/active/ssl_experiments/future_prediction_ssl`
- `analysis/active/ssl_experiments/cross_trained_mamba`
- `analysis/active/ssl_experiments/timestep_flexible_ssm`
- `analysis/reference/possm/possm_ssl` as reference/evidence
- `analysis/active/transfer_benchmark/ssl_autoresearch` as the S5/reference
  benchmark scaffold

The local `external/speechBCI` checkout was removed during pruning. Use the
published Stanford/Card repository or the archived notes when the reference
recipe needs to be checked again.

Treat `masked_ssl` and `contrastive_ssl` as legacy code, but do not delete them
until shared cache/probe imports have been migrated into `ssl_core`.

## Data Audit

Repo-local data status as of this audit:

- `data/cache_v1/brain2text24` is a duplicate of
  `/Users/home/thesis/data/cache_v1/brain2text24`.
  - `metadata.json` hash: identical
  - `manifest.jsonl` hash: identical
  - content hash ignoring `.DS_Store`: identical
  - external cache has one extra ignored `.DS_Store` in `shards/`
- `data/cache_v1` is not a duplicate of the full canonical cache.
  - repo-local: `brain2text24` only, about `2.3G`
  - `/Users/home/thesis/data/cache_v1`: multiple datasets, about `19G`
- `data/cache_v1_smoothed_sigma2p0` was copied to
  `/Users/home/thesis/data/cache_v1_smoothed_sigma2p0` and verified by content
  hash.
- repo-local `data/stats` was merged into `/Users/home/thesis/data/stats`
  without overwriting existing files.

Data cleanup steps:

- [x] Copy or regenerate `data/cache_v1_smoothed_sigma2p0` under
  `/Users/home/thesis/data/cache_v1_smoothed_sigma2p0`.
- [x] Verify the moved smoothed cache with manifest, metadata, shard count, and
  a content hash.
- [x] Merge any repo-local stats that are missing from
  `/Users/home/thesis/data/stats`.
- [x] Update `docs/notes/cache_and_stats_inventory.md` if the canonical local
  smoothed cache path changes.
- [x] Remove repo-local `data/cache_v1/brain2text24` after confirming scripts
  point at `/Users/home/thesis/data/cache_v1`.
- [x] Remove repo-local `data/cache_v1_smoothed_sigma2p0` only after the new
  external copy is verified.

## Safe Local Clutter

These are safe cleanup candidates because they are generated, ignored, or local
scratch artifacts:

- [x] remove `__pycache__/` directories
- [x] remove `.DS_Store` files
- [x] remove `.tmp_willett_s5_tx_sbp_seed7_60k_curve.png`
- [x] decide whether to keep or move `mixedbins.png`
- [x] move or delete `willett plots/`
- [x] remove `tmp/` after checking for unrecovered notes
- [x] add `.venv/` to `.gitignore`
- [x] remove or relocate repo-local `.venv/`
- [x] remove or relocate top-level `ssh_key` and `ssh_key.pub`

## Git State Cleanup

The working tree currently contains a mixture of real source changes, archived
notebook moves, new active experiment packages, and local artifacts. Do not use
bulk reset commands.

- [ ] Stage the intentional archive move from active `s5_*` notebooks to
  `analysis/active/ssl_experiments/archive/notebooks/`.
- [ ] Keep `analysis/active/ssl_experiments/README.md` and `AGENTS.md`; they
  describe the current cleanup direction.
- [ ] Review whether `abstract/abstract-final.md` belongs in this repo.
- [ ] Review whether `get_started.py` is useful or scratch.
- [ ] Keep `scripts/modal/` because it is referenced by the Modal workflow note.
- [ ] Keep `scripts/runpod/` because it is referenced by the RunPod runbook.
- [ ] Keep `docs/paper_notes/BIT.pdf` only if local PDF access is worth the
  extra binary file; otherwise rely on notes and a source URL.

## Archive Or Move Out Of Active Tree

These are not first-line active work, but some are useful provenance:

- [x] Move `analysis/active/channel_ablation` out of the repo tree or into an
  external archive. Its conclusion is already recorded in
  `docs/notes/archive/channel_ablation_summary.md`.
- [ ] Keep `docs/notes/archive/` as the historical ledger.
- [ ] Keep `analysis/active/ssl_experiments/archive/` for superseded notebooks
  whose conclusions are already captured.
- [ ] Consider moving old active notebooks into `archive/notebooks/` once their
  conclusions are captured in notes.

## Codebase Prune Plan

Do this after local clutter and data relocation:

- [ ] Move reusable cache helpers from `masked_ssl.cache` into `ssl_core.cache`.
- [ ] Move reusable CTC/probe dataset helpers from `masked_ssl.probe` into
  `ssl_core.ctc` or a new `ssl_core.probe`.
- [ ] Update imports in active packages to depend on `ssl_core` rather than
  `masked_ssl`.
- [ ] Run focused tests for `ssl_core`, `ssm_ssl`, `willett_reconstruction`,
  `future_prediction_ssl`, and `cross_trained_mamba`.
- [ ] Archive or delete `contrastive_ssl` once no active imports or notes depend
  on it.
- [ ] Split `masked_ssl` into either a small archived package or delete it after
  its reusable utilities are migrated.
- [ ] Keep POSSM code until POSSM stage-2 temporal-patching follow-up is either
  completed or explicitly abandoned.

## Verification Before Final Prune

- [ ] `git status --short` contains no accidental local artifacts.
- [ ] `git ls-files --others --exclude-standard` contains only intentional new
  source/docs.
- [ ] active unit tests pass.
- [ ] cache audit scripts still point to `/Users/home/thesis/data/...` for
  canonical local data.
- [ ] notebooks and run scripts no longer rely on repo-local `data/`.
- [ ] README and AGENTS files describe the final active layout.
