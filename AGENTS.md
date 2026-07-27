# Objective

This folder is focused on building a self-supervised learning pipeline for Utah array data that can decode speech motor cortex activity into speech intention.

# Project Context

Experiment summaries and architecture notes live in `docs/notes`, including what has already been tried and what still needs exploration.

Notes on relevant published papers live in `docs/paper_notes`.

The dataset for this work exists both locally and on Google Drive. Reusable cache
artifacts belong under the Drive/local data roots, not experiment-output folders.
For POSSM/SSL normalization artifacts, prefer the organized stats layout under
`utah_ssl/data/stats` when available.

For local cache work, treat `/Users/home/thesis/data/cache_v1` as the primary
full-corpus cache root for BIT-style and multi-dataset SSL preparation. The
repo-local cache at `/Users/home/thesis/utah-ssl/data/cache_v1` may contain
only a partial subset for lightweight local experiments, so do not assume it is
the canonical full dataset inventory.

A short inventory of the important cache roots and normalization-stat artifacts
lives in `docs/notes/cache_and_stats_inventory.md`. Check it before changing
notebook cache paths or normalization settings.

For the current Modal-based BIT stage-1 workflow, including persistent volume
names, expected in-volume filesystem layout, archive upload strategy, and the
exact helper scripts / commands used, see
`docs/notes/modal_bit_stage1_workflow.md`.

The active generic SSM SSL path is in
`analysis/active/ssl_experiments/ssm_ssl`, with shared experiment utilities in
`analysis/active/ssl_experiments/ssl_core`.

The POSSM reference implementation is in
`analysis/reference/possm/possm_ssl` and is best treated as a reference
implementation and evidence source. Its notebooks live under
`analysis/reference/possm/notebooks`.

The Willett speechBCI reference code is not kept in this repo anymore. Use the
published Stanford/Card repository or archived notes for architecture and
training-recipe comparisons.
