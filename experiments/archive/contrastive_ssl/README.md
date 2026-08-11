# Archived Contrastive SSL

This branch tested future-prediction and augmentation-based InfoNCE objectives
with S5 encoders. It was deprioritized because high retrieval scores were
dominated by shortcut-prone session/shard structure and did not translate into
strong phoneme decoding.

- Environment: Colab-oriented PyTorch/S5 notebooks.
- Entry points: the notebooks under `notebooks/` and package modules in this
  directory, run from the repository root.
- Results: [results/README.md](results/README.md).
- Artifacts: established Drive paths under
  `/content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/`.
