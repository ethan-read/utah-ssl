# ssl_autoresearch

This directory is the full, platform-neutral SSL autoresearch scaffold.

## Purpose

- separate the real CUDA-ready benchmark from the local smoke-test folder
- keep benchmark contract code fixed while model code evolves
- make the project portable across laptop, Colab, and cluster environments

## Editing Rules

- treat `prepare.py`, `data.py`, `run_experiment.py`, and `search_space.py` as benchmark infrastructure
- treat `train.py` as the primary editable surface for architecture and objective work
- if the benchmark contract changes, do not compare new runs against older incompatible `results.tsv` rows

## Artifact Policy

- generated artifacts should go under `outputs/transfer_benchmark/ssl_autoresearch`
- keep source directories readable and light enough to push to GitHub cleanly
- prefer explicit run directories and stable checkpoint names over ad hoc files

## Checkpoint Policy

- retain multiple pretrained encoder versions when they correspond to distinct pretraining corpora, objective families, or benchmark variants
- never overwrite the only copy of a pretrained encoder that may be needed for ablations
- checkpoint names should make the following legible:
  - dataset family
  - backbone
  - objective family
  - patching condition
  - split / benchmark variant

## Benchmark Discipline

- the benchmark loop is currently frozen around held-out-session `session_avg_val_bpphone`
- avoid comparing heterogeneous SSL training losses directly once multiple objective families are in scope
- prefer a fixed downstream metric over objective-native losses for keep / discard decisions
- treat the benchmark phoneme head as common infrastructure, not as an optional per-objective add-on
- distinguish clearly between:
  - probe training, which happens in every benchmark regime
  - encoder adaptation budget, which defines the regime
- keep the transfer regimes explicit:
  - `A`: target-day normalization only
  - `B1`: affine-only encoder-side adaptation
  - `B2`: full-encoder fine-tuning
- do not let `B2` become the default inner-loop metric; it is a secondary comparison because fast adaptation is the main practical goal
- treat `s5 + future_prediction` as the current reference benchmark path
- keep `debug_gru` only as a bootstrap fallback
- add `Mamba` and later S5-specific internal search inside `train.py` without changing the held-out-session probe contract
