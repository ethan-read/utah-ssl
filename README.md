# utah-ssl

Subset repository for the Utah-array SSL / transfer benchmark work.

For the current interpretation of experiment results, start with
[`docs/notes/experiment_synthesis.md`](docs/notes/experiment_synthesis.md).

## Included

- `analysis/active/transfer_benchmark/ssl_autoresearch`
  - full benchmark scaffold
  - canonical pure-PyTorch `s5` reference backbone
  - held-out-session phoneme probe benchmark
- `analysis/active/transfer_benchmark/ssl_autoresearch_local`
  - small local smoke-test harness
- `docs/notes/experiment_synthesis.md`
  - current canonical synthesis of experiment results and active conclusions
- `docs/notes/ssl_architecture_choices.md`
  - current architecture decision log
- `analysis/active/ssl_experiments/ssl_core`
  - shared cache, normalization-stat, CTC, temporal patching, and reporting helpers
- `analysis/active/ssl_experiments/ssm_ssl`
  - active generic `S5`/`Mamba` SSL experiments with downstream CTC controls
- `analysis/reference/possm/possm_ssl`
  - POSSM reconstruction and phoneme fine-tuning reference implementation
- `docs/paper_notes/`
  - paper architecture notes that informed the benchmark design

## Not Included

- raw datasets
- cached neural features
- generated outputs, checkpoints, or logs
- unrelated thesis material outside the SSL / autoresearch work

## Data / Outputs

The full scaffold expects data and outputs to live outside the repo and be routed through environment variables.

For active SSL experiment notebooks and scripts, reusable Drive artifacts are organized under:

- `/content/drive/MyDrive/utah_ssl/data/`
  - cache roots such as `cache_v1` and `cache_v1_smoothed_sigma2p0`
  - reusable normalization stats under `data/stats`
- `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/`
  - experiment runs, checkpoints, logs, and plots

See [`docs/notes/cache_and_stats_inventory.md`](docs/notes/cache_and_stats_inventory.md) before changing cache roots or normalization-stat paths.

The main ones are:

- `SSL_AUTORESEARCH_OUTPUT_ROOT`
- `SSL_AUTORESEARCH_TX_CACHE_DIR`
- `SSL_AUTORESEARCH_SBP_CACHE_DIR`
- `SSL_AUTORESEARCH_B2T25_ROOT`
- `SSL_AUTORESEARCH_B2T25_HDF5_ROOT`

See [`analysis/active/transfer_benchmark/ssl_autoresearch/README.md`](analysis/active/transfer_benchmark/ssl_autoresearch/README.md) for the current benchmark contract.
