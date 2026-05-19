# Objective

This folder is focused on building a self-supervised learning pipeline for Utah array data that can decode speech motor cortex activity into speech intention.

# Project Context

Experiment summaries and architecture notes live in `docs/notes`, including what has already been tried and what still needs exploration.

Notes on relevant published papers live in `docs/paper_notes`.

The dataset for this work exists both locally and on Google Drive. Reusable cache
artifacts belong under the Drive/local data roots, not experiment-output folders.
For POSSM/SSL normalization artifacts, prefer the organized stats layout under
`utah_ssl/data/stats` when available.

A short inventory of the important cache roots and normalization-stat artifacts
lives in `docs/notes/cache_and_stats_inventory.md`. Check it before changing
notebook cache paths or normalization settings.

The active POSSM implementation is in
`analysis/active/ssl_experiments/possm_ssl`. POSSM sweep/launcher scripts live
under `analysis/active/ssl_experiments/possm_ssl/scripts`; top-level
`analysis/active/ssl_experiments/possm_stage*.py` files are compatibility
wrappers.

The Willett speechBCI reference code is cloned locally at `external/speechBCI` for architecture and training-recipe comparisons.
