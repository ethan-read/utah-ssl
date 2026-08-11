# BIT-Style Experiments

This active branch tests specific BIT-derived pretraining ideas on Utah-array
speech data. It is a paper-inspired adaptation, not an official BIT
implementation.

The current code supports broad multi-dataset Stage 1 pretraining and matched
Brain-to-Text 2024 CTC evaluation. Reusable data and evaluation contracts live
in `utah_ssl`; branch-specific objectives and training remain here.

- Current design: [design/README.md](design/README.md)
- Detailed BIT-to-S5 fidelity guide:
  [design/faithful_bit_to_s5_adaptation.md](design/faithful_bit_to_s5_adaptation.md)
- Modal launchers: `launchers/modal/`
- Scripted entry point: `python -m experiments.bit_style.scripts.run_generic_ssm_ssl`
- Results: [results/README.md](results/README.md)

Colab remains the default interactive workflow. Substantial existing BIT runs
use the documented Modal launchers.
