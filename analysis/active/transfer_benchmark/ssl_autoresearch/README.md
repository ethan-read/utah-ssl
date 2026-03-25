# SSL Autoresearch

This directory is the fuller SSL autoresearch scaffold intended to replace the local smoke-test setup.

It is designed to be:

- CUDA-ready
- portable across multiple runtime environments
- clean to push to GitHub
- strict about separating benchmark infrastructure from model edits

## Path Handling

The scaffold is set up so code paths are stable but data paths are relocatable.

The rule is:

- code lives in the GitHub repo
- data and outputs may live anywhere
- generated manifests should store `root_key + relative path`, not machine-specific absolute data paths

Current environment-variable overrides:

- `SSL_AUTORESEARCH_OUTPUT_ROOT`
- `SSL_AUTORESEARCH_TX_CACHE_DIR`
- `SSL_AUTORESEARCH_SBP_CACHE_DIR`
- `SSL_AUTORESEARCH_B2T25_ROOT`
- `SSL_AUTORESEARCH_B2T25_HDF5_ROOT`

This is intended to work cleanly for:

- local workstation paths
- Google Drive mounts in Colab
- rented GPU nodes with attached storage

## Current Status

What is already set up:

- runtime profiles for local, single-GPU, Colab, and cluster-style launches
- output/checkpoint/run directory management under `outputs/`
- Brain2Text25 cache inventory and source discovery
- relocatable path-root metadata for generated inventories and manifests
- a generic experiment runner and TSV logger
- an explicit bootstrap search-space module
- a runnable benchmark loop in `train.py`:
  - SSL pretraining on source sessions
  - held-out-session phoneme probes on target sessions
  - support for adaptation regimes `A`, `B1`, and `B2`
  - primary metric `session_avg_val_bpphone`
  - runnable backbones:
    - `debug_gru` bootstrap baseline
    - canonical pure-PyTorch `s5` reference backbone

What is intentionally still incomplete:

- `Mamba` integration
- the wider SSL objective family matrix beyond `future_prediction`
- S5-specific internal architecture search
- any large-scale GPU tuning beyond smoke-tested bootstrap settings

That is deliberate. The benchmark contract is now real enough to run, and `s5` is now the first real reference backbone. The remaining bootstrap pieces should still be treated as plumbing and metric-validation infrastructure rather than the final thesis model family.

## Files

- [`prepare.py`](prepare.py): frozen runtime profiles, artifact paths, summary formatting, and benchmark metadata
- [`data.py`](data.py): cache inventory, session splitting, and source discovery utilities
- [`build_inventory.py`](build_inventory.py): writes a JSON snapshot of the current data inventory
- [`build_probe_manifest.py`](build_probe_manifest.py): writes an utterance-level labeled Brain2Text25 manifest for the future phoneme probe
- [`train.py`](train.py): editable benchmark/training entry point
- [`run_experiment.py`](run_experiment.py): one-run wrapper and generic result logger
- [`search_space.py`](search_space.py): current allowed architecture/objective choices
- [`program.md`](program.md): agent instructions for the eventual autoresearch loop

## Runtime Profiles

Profiles are defined in [`prepare.py`](prepare.py) and are meant to capture environment differences without changing source code.

Current profiles:

- `local_debug`
- `single_gpu`
- `colab_cuda`
- `cluster_cuda`

The profile controls things like:

- device priority
- dataloader worker count
- `pin_memory`
- preferred batch-size hint
- pretraining and probe time budgets
- whether `torch.compile` should even be considered

## Artifact Layout

All generated files are routed under:

- `outputs/transfer_benchmark/ssl_autoresearch/`

Subdirectories are created automatically for:

- `checkpoints/`
- `runs/`
- `logs/`
- `inventories/`
- `manifests/`

This keeps the active source tree clean for GitHub and makes it easier to move between platforms.

## Current Benchmark Contract

The benchmark center is currently:

- a fixed-budget causal phoneme probe
- evaluated on held-out sessions
- using session-averaged validation phoneme loss as the primary comparison metric

The most likely metric form is:

- `val_bpphone`

meaning validation negative log-likelihood in bits per target phoneme, averaged equally across held-out sessions.

This metric is shared across benchmark regimes, so raw SSL losses are not used for keep / discard decisions.

## Adaptation Regimes

The current transfer comparison is:

- `A`: target-day normalization only + phoneme head
- `B1`: target-day normalization + affine-only input adaptation + phoneme head
- `B2`: target-day normalization + full-encoder fine-tuning + phoneme head

Important interpretation:

- the phoneme head is trained in all regimes
- what changes between regimes is the encoder-side adaptation budget
- `B2` is a secondary comparison, not the main inner-loop benchmark, because the thesis goal is a model that is useful under quick adaptation

`A` is currently the cheapest common benchmark regime and the default inner-loop candidate.

## Bootstrap Search Surface

The benchmark loop is runnable today with:

- `backbone = s5`
- `objective_family = future_prediction`
- `adaptation_regime in {A, B1, B2}`
- `patch_size in {1, 3, 5}`
- `patch_stride in {1, 3, 5}` with `patch_stride <= patch_size`
- `standardize_scope in {subject, session}`
- `post_proj_norm in {none, rms}`

For patched variants, the encoder uses only stride-aligned valid patches. Trailing bins that do not fit the chosen `(patch_size, stride)` schedule are intentionally dropped rather than padded into an extra terminal patch.

`debug_gru` remains available only as a bootstrap fallback. The active benchmark-facing search surface now starts from the canonical `s5` reference backbone.

## S5 Integration Policy

The next backbone implementation pass should start with one canonical pure-PyTorch `S5` reference block.

- first pass:
  - actual `S5`, not a silent `S4D` fallback
  - same benchmark contract and sequence interface as the bootstrap path
  - one stable reference block before opening internal architecture search
- later pass:
  - allow bounded `S5`-specific autoresearch over choices such as norm type, feedforward width, and residual / skip style

The point of this staging is to separate “does `S5` work in this benchmark at all?” from “which `S5` block variant is best?”

## Design Principle

This directory should now be treated as the active benchmark surface.

The next major implementation step is extending the current `s5` reference path with:

- `Mamba`
- more SSL objective families
- later S5-specific internal architecture search

while preserving the same held-out-session benchmark contract.
