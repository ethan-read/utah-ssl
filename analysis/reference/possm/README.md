# POSSM reference implementation

This folder contains the POSSM reconstruction and phoneme fine-tuning work
retained as a reference implementation and evidence source. It is intentionally
separate from the active generic SSL experiments.

- `possm_ssl/`: reusable POSSM models, training, fine-tuning, reporting, tests,
  and sweep launchers.
- `notebooks/s6_possm_maskedreconstruction.ipynb`: original brain2text24
  reconstruction and transfer workflow.
- `notebooks/s13_brain2text25_long_pretraining.ipynb`: Colab workflow for the
  longer brain2text25 reconstruction and POSSM-GRU transfer comparison.
- `notebooks/s14_possm_pooled_pretraining.ipynb`: optimized pooled
  Brain2Text24/25 Stage-1 workflow and compute-matched Brain2Text24 transfer.
- `notebooks/s15_possm_pooled_cache_preparation.ipynb`: one-time Colab build
  and validation of Brain2Text25-only area-6v/TX-FP16 caches, plus mixed-root
  pooled stats and sampling benchmarks.
- `EXPERIMENT_NOTES.md`: current experiment question, Colab recovery guidance,
  cache workflow, pitfalls, and the planned Stage-2 speedup idea.

From the repository root, add `analysis/reference/possm` to `PYTHONPATH` before
running the package or its launcher scripts so that `import possm_ssl` resolves.
