# Archived Mamba Future Prediction

This branch adapted a causal forecasting objective to Brain-to-Text 2024. It
was deprioritized because the explored future-prediction path did not establish
more decodable latent states than matched controls, while Mamba was less
promising than recurrent baselines for the current project.

- Environment: the retained Modal image definition, including optional Mamba
  CUDA kernels, is the last-known runtime.
- Entry points: `launchers/modal/` and the package tests.
- Results: [results/README.md](results/README.md).
- Artifacts: Modal/Drive paths remain encoded in the launchers and result log.
