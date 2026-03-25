# ssl_autoresearch_local

This directory is a local staging area for an autoresearch-style SSL benchmark.

## Purpose

- keep the local benchmark cheap, fixed, and auditable
- validate benchmark mechanics before building the larger CUDA-dependent pipeline
- treat this folder as active thesis work, not legacy experimentation

## Editing Rules

- `prepare.py`, `data.py`, and `run_experiment.py` define the benchmark contract and should remain fixed during experiment loops
- `train.py` is the editable surface for model and objective changes
- if the benchmark contract changes, reset or clearly separate prior `results.tsv` records so incomparable runs are not mixed

## Checkpoint Policy

- retain multiple pretrained encoder versions when they correspond to meaningfully different pretraining corpora or pretraining strategies
- do not overwrite the only copy of a pretrained model if it may later be needed for ablations such as:
  - pretraining data included versus excluded
  - SSL objective A versus SSL objective B
  - same-session versus held-out-session benchmark variants
- checkpoint naming should make the pretraining condition explicit so later transfer comparisons are reproducible

## Research Intent

- the local loop is for fast comparison, not for final thesis claims
- once a candidate looks promising locally, it should be promoted into the fuller benchmark rather than endlessly optimized here
