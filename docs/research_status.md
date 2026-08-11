# Research Status

The project is exploring improvements to Brain-to-Text 2024 and 2025 speech
decoding without committing to one mechanism prematurely. Transfer learning
from other Utah-array datasets is the most promising broad direction. BIT-style
and POSSM-style ideas are active concrete branches; manifold analysis remains
exploratory.

## Current Findings

- The Willett-derived supervised recipe remains the decoding anchor.
- Supervised S5 reached the strongest recorded local Brain-to-Text 2024 result,
  with best observed validation PER `0.25591` for the area-6v TX+SBP run.
- POSSM-style reconstruction pretraining improves optimization over matched
  random initialization, but remains behind the best supervised decoder. The
  strongest recorded pooled Brain-to-Text 2024/2025 SBP run reached a best
  observed validation PER of `0.377203`.
- Existing contrastive, masked-reconstruction, forecasting, and generic SSM
  experiments did not establish a stronger decoding path and are archived.
- GRU hidden states make model beliefs substantially more linearly accessible
  than raw input windows, but the manifold work does not yet support a broader
  biological or dynamical claim.

## Canonical Evidence

- [Supervised baselines](../experiments/supervised_baselines/results/README.md)
- [BIT-style work](../experiments/bit_style/results/README.md)
- [POSSM-style work](../experiments/possm_style/results/README.md)
- [Manifold analyses](../experiments/manifolds/results/README.md)
- [Archived experiments](../experiments/archive/README.md)

Detailed historical interpretations from before the reorganization are
preserved in the generic SSM archive's experiment-synthesis snapshot.

Headline PER values in this status page are the lowest observed validation PER
within their respective runs. They are useful run summaries, not held-out test
estimates; differences in splits, budgets, and mostly single-seed evidence
limit cross-run ranking claims.
