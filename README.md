# utah-ssl

Subset repository for the Utah-array SSL / transfer benchmark work.

## Included

- `analysis/active/transfer_benchmark/ssl_autoresearch`
  - full benchmark scaffold
  - canonical pure-PyTorch `s5` reference backbone
  - held-out-session phoneme probe benchmark
- `analysis/active/transfer_benchmark/ssl_autoresearch_local`
  - small local smoke-test harness
- `docs/notes/ssl_architecture_choices.md`
  - current architecture decision log
- `docs/paper_notes/`
  - paper architecture notes that informed the benchmark design

## Not Included

- raw datasets
- cached neural features
- generated outputs, checkpoints, or logs
- unrelated thesis material outside the SSL / autoresearch work

## Data / Outputs

The full scaffold expects data and outputs to live outside the repo and be routed through environment variables.

The main ones are:

- `SSL_AUTORESEARCH_OUTPUT_ROOT`
- `SSL_AUTORESEARCH_TX_CACHE_DIR`
- `SSL_AUTORESEARCH_SBP_CACHE_DIR`
- `SSL_AUTORESEARCH_B2T25_ROOT`
- `SSL_AUTORESEARCH_B2T25_HDF5_ROOT`

See [`analysis/active/transfer_benchmark/ssl_autoresearch/README.md`](analysis/active/transfer_benchmark/ssl_autoresearch/README.md) for the current benchmark contract.
