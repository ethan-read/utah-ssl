# Archived Masked Reconstruction SSL

This branch tested causal masked reconstruction and MAE-style objectives with
S5 encoders. It was deprioritized after reconstruction improvements failed to
produce competitive downstream phoneme representations.

- Environment: Colab-oriented PyTorch with the shared `utah_ssl` cache and S5
  backbones.
- Entry points: notebooks under `notebooks/`; tests run with
  `python -m unittest discover -s experiments/archive/masked_reconstruction_ssl/tests`.
- Results: [results/s5_masked_reconstruction.md](results/s5_masked_reconstruction.md).
- Artifacts: Drive output roots recorded in the notebooks and result report.
