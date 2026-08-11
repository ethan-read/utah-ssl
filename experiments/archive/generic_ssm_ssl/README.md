# Archived Generic SSM SSL

This frozen branch preserves the generic S5/Mamba SSL backend, tests, launchers,
and historical notebooks that existed before the active BIT path was isolated.
It was deprioritized because generic SSM objectives had not yielded a useful
phoneme basis and recurrent transfer work became the stronger direction.

- Environment: PyTorch with optional Mamba CUDA kernels; Colab and Modal
  launchers are retained.
- Entry point: `python -m experiments.archive.generic_ssm_ssl.scripts.run_generic_ssm_ssl`.
- Results: `results/`, including the complete pre-reorganization synthesis.
- Historical BIT codebase audit:
  `design/bit_to_s5_codebase_audit.md`.
- Artifacts: Drive and Modal output locations remain recorded in the retained
  notes and launchers.
