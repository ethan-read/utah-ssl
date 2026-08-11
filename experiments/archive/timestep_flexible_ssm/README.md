# Timestep-Flexible S5 Decoder

> **Archived.** The branch was deprioritized with the broader SSM direction.
> Its code, tests, Colab notebooks, and result ledger remain runnable from the
> repository root. The last-known environment is the Colab/PyTorch setup stored
> in `notebooks/timestep_flexible_s5.ipynb`; artifacts remain under the Drive
> output root recorded there.

This package trains a Willett-style supervised phoneme decoder with:

- canonical Brain2Text24 phoneme labels
- train/eval bin-size configuration
- rebinned feature views
- millisecond-based patch and smoothing config
- a fixed patched-token timebase across `20 ms` and `40 ms` evaluation views

The first intended comparison is a model trained at `20 ms` and evaluated at:

- `20 ms`
- `40 ms`

Experiment-specific artifacts live alongside the package:

- notebook: `notebooks/timestep_flexible_s5.ipynb`
- running log: `tests_and_results.md`
