# Setup and verification

## Supported workflow

Colab with Google Drive is the supported environment for complete training
runs. Local setup is intended for unit tests, documentation work, lightweight
analysis, and debugging before sending a run to Colab, Modal, or RunPod.

Use Python 3.10. Run commands from the repository root; the repository is not
currently distributed as an installable Python package, and no legacy
`PYTHONPATH` entries are required.

## Lightweight local environment

Create an isolated Python 3.10 environment using your preferred environment
manager, then install the libraries needed by the reusable package and active
test suites:

```bash
python -m pip install numpy pandas torch matplotlib scikit-learn transformers
```

This is a lightweight compatibility set, not a lockfile for GPU training.
Platform-specific PyTorch and CUDA packages should follow the requirements in
the relevant notebook or launcher.

Additional optional dependencies are scoped to the functionality that uses
them:

- `modal` for Modal launchers;
- `optuna` for POSSM-style hyperparameter searches;
- `tensorflow-cpu`, `pyyaml`, and `tfrecord` for released Willett checkpoint
  conversion and TFRecord evaluation;
- `mamba-ssm` and `causal-conv1d` for optimized Mamba GPU kernels.

Do not install optional GPU kernels merely to run the ordinary S5, GRU, POSSM,
or manifold tests. Exact remote package versions and CUDA-image requirements
belong in the owning experiment's launcher documentation.

## Active test suites

Run the reusable-core and active-branch tests from the repository root:

```bash
python -m unittest discover -s utah_ssl/tests -p 'test*.py' -q
python -m unittest discover -s experiments/supervised_baselines/tests -p 'test*.py' -q
python -m unittest discover -s experiments/bit_style/tests -p 'test*.py' -q
python -m unittest discover -s experiments/possm_style/tests -p 'test*.py' -q
python -m unittest discover -s experiments/manifolds/tests -p 'test*.py' -q
```

Archived branches retain their own tests and last-known environments. Run
those suites only when inspecting or restarting the corresponding branch.

## Documentation and notebook checks

Before committing structural or notebook changes:

```bash
find . -name '*.ipynb' -type f -print0 | xargs -0 -n1 jq empty
git diff --check
```

The notebook check requires `jq`. It validates notebook JSON without executing
cells or changing saved outputs.

## Colab and remote training

Follow [the Colab workflow](../workflows/colab/README.md) for the default
interactive setup. Notebooks live with their experiment branch and are the
canonical executable entry points. Modal and RunPod package versions are
documented with the launchers that use them rather than imposed on the local
environment.
