# POSSM-Style Experiments

This active branch contains a paper-derived implementation of POSSM-style
reconstruction pretraining and phoneme fine-tuning. POSSM code has not been
released, so this is neither an official implementation nor a verified
reproduction; architectural details were inferred from the paper descriptions.

## Colab Notebooks

- [Brain-to-Text 2024 reconstruction and transfer](notebooks/brain2text24_masked_reconstruction_transfer.ipynb)
- [Brain-to-Text 2025 long pretraining](notebooks/brain2text25_long_pretraining.ipynb)
- [Pooled Brain-to-Text 2024/2025 pretraining](notebooks/pooled_brain2text24_brain2text25_pretraining.ipynb)
- [Pooled cache preparation](notebooks/pooled_cache_preparation.ipynb)

Run Python entry points from the repository root using imports under
`experiments.possm_style`. Current results are indexed in
[results/README.md](results/README.md); operational recovery details remain in
[design/implementation_notes.md](design/implementation_notes.md).
