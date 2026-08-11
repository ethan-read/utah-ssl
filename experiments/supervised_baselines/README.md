# Supervised Baselines

This branch contains the shared supervised phoneme-decoding implementation and
separate GRU, S5, and S4D baseline experiments. It anchors split,
normalization, smoothing, temporal-patching, CTC, and reporting decisions used
elsewhere in the repository.

## Notebooks

- [Willett GRU baseline](notebooks/willett_gru_baseline.ipynb)
- [Supervised S5 baseline](notebooks/supervised_s5_baseline.ipynb)
- [Supervised S4D baseline](notebooks/supervised_s4d_baseline.ipynb)

Run the scripted trainer from the repository root:

```bash
python -m experiments.supervised_baselines.train --help
```

RunPod launchers and their environment are under `launchers/runpod/`. Results
are indexed in [results/README.md](results/README.md).

The GRU includes an LLM-assisted Python port/adaptation with unresolved
upstream source and licensing details. See [PROVENANCE.md](PROVENANCE.md) before
including this branch in a public release.
