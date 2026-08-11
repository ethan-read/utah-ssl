# Objective

This repository explores methods for improving Utah-array speech decoding on
Brain-to-Text 2024 and 2025. Transfer learning across Utah-array datasets is a
primary direction, with active BIT-style, POSSM-style, supervised-baseline, and
manifold-analysis branches.

# Repository Structure

- Reusable code lives in `utah_ssl`.
- Active experiments live in `experiments/supervised_baselines`,
  `experiments/bit_style`, `experiments/possm_style`, and
  `experiments/manifolds`.
- Inactive work lives under `experiments/archive` and must not be imported by
  active code.
- Cross-cutting status and data documentation lives in `docs`; detailed
  reasoning and results belong to the experiment that produced them.
- Paper notes live in `docs/paper_notes`.

POSSM code has not been released. Treat `experiments/possm_style` as a
paper-derived implementation, not an official reference implementation or a
verified reproduction.

The GRU in `experiments/supervised_baselines` includes an LLM-assisted Python
port/adaptation of likely TensorFlow source. Preserve its `PROVENANCE.md` and
review upstream source and licensing before public distribution.

# Workflow

Colab is the default workflow, using branch-owned notebooks and the shared
conventions in `workflows/colab`. The Colab CLI may supplement notebooks when
its installed version has been verified. Shared Modal and RunPod infrastructure
lives under `workflows`; experiment launchers remain with their branch.

Run commands from the repository root. Use canonical imports under `utah_ssl.*`
and `experiments.*`; do not recreate the former `analysis.*` import hierarchy.

# Data and Artifacts

The dataset exists locally and on Google Drive. Reusable caches and
normalization statistics belong under the Drive/local data roots, not
experiment-output folders. Read `docs/data/cache_and_stats_inventory.md` before
changing cache paths, feature modes, or normalization settings.

The local full-corpus cache root is `/Users/home/thesis/data/cache_v1`. The
repo-local data directory, if present, may contain only a lightweight subset
and is not the canonical inventory. The usual synced Drive root is
`/Users/home/My Drive/utah_ssl`; Colab uses
`/content/drive/MyDrive/utah_ssl`.

Brain-to-Text 2024 and 2025 always use the first 128 area-6v channels. Prefer
SBP for Brain-to-Text decoding and use the validated clipped-FP16 SBP cache
variants; TX remains the common signal for auxiliary datasets that do not
provide SBP.

Preserve existing Drive and remote artifact paths when reorganizing source.
Archived branches must retain their last-known commands, environment, result
ledger, and artifact locations, without adding criteria for reopening them.
