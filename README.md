# Utah SSL

Research code for improving Utah-array speech decoding across Brain-to-Text
datasets. The repository is organized around reusable infrastructure and
specific experiment ideas rather than the order in which work was attempted.

## Start Here

- [Research status](docs/research_status.md): current high-level findings.
- [Setup and verification](docs/setup.md): supported environments,
  dependencies, and test commands.
- [Data documentation](docs/data/cache_and_stats_inventory.md): canonical cache
  roots, signal layouts, and normalization artifacts.
- [Colab workflow](workflows/colab/README.md): the default execution workflow.
- [Experiments](experiments/README.md): active and archived branches.

## Layout

- `utah_ssl/`: reusable cache, dataset, model, normalization, CTC, patching,
  reporting, and data-maintenance code.
- `experiments/supervised_baselines/`: Willett-derived GRU and supervised S5/S4D
  comparisons.
- `experiments/bit_style/`: active BIT-style pretraining and transfer work.
- `experiments/possm_style/`: paper-derived POSSM-style experiments.
- `experiments/manifolds/`: exploratory representation and trajectory analyses.
- `experiments/archive/`: inactive branches retained in restartable form.
- `workflows/`: shared Colab, Modal, and RunPod execution infrastructure.
- `docs/`: cross-cutting data, paper, and research-status documentation.

Run Python commands from the repository root. Canonical imports use
`utah_ssl.*` and `experiments.<branch>.*`; the old `analysis.*` import paths are
not supported.

## Data and Outputs

Datasets, reusable caches, normalization statistics, checkpoints, and generated
outputs are not stored in the repository. Colab uses
`/content/drive/MyDrive/utah_ssl`; the canonical local cache is normally
`/Users/home/thesis/data/cache_v1`.

Existing Drive artifact paths remain stable across this source-code
reorganization. New runs use the documented
[artifact layout](docs/run_artifact_layout.md); existing outputs are not moved
or renamed to conform to it. Generated configurations, logs, metrics,
checkpoints, and plots remain in persistent artifact storage rather than being
committed here. Google Drive is the permanent source of record for new run
artifacts; public artifact hosting is not required.
