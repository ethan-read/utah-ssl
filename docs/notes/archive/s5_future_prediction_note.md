# S5 Future Prediction Note

This note records the current status of the `S5` future-prediction line of SSL experiments.

## Earlier Result

The original direct next-step prediction runs were not promising:

- `S5` did not beat the trivial `predict 0` baseline in our early future-prediction runs.
- Based on those runs alone, future prediction did not look useful with `S5`.

## Current Update: Contrastive Future Prediction

The newer `S5` future-prediction experiments use a contrastive patch-level objective (`future_infonce`) rather than direct next-bin regression.

Current rough workflow:

1. Pretrain a causal `S5` encoder on non-`Brain2Text25` data with future-token `InfoNCE`.
2. Inspect the learned embedding geometry and nuisance structure.
3. Test transfer with a cheap held-out `Brain2Text25` phoneme `CTC` probe.

## Main Findings So Far

- raw SSL retrieval metrics can become extremely high, including runs near `99%` top-1, but this is not reliable evidence of useful phoneme structure
- per-horizon retrieval remains much weaker at longer horizons:
  - horizon `4`: top-1 about `0.127` against about `864` candidates
  - horizon `8`: top-1 about `0.036` against about `736` candidates
- at horizon `8`, performance is close to the within-segment shortcut level (`1 / 23 ~= 0.043`), suggesting the model may often identify the correct segment without learning strong longer-range temporal alignment
- the learned embeddings are consistently low-rank / anisotropic rather than healthy, broad representations:
  - effective rank only about `12.7` to `14.5` in a `256`-dimensional space
  - top `5` principal components explain about `65%` to `71%` of the variance
- nuisance structure is present but the strongest effect is very local organization:
  - `shard_within_session_nn` stays around `0.96` to `0.97`
  - this suggests the representation is strongly grouped by shard-local recording context within a session
- downstream phoneme probing does not yet support a strong claim that the objective learns phoneme-discriminative structure
  - the frozen SSL probe improves some `CTC` metrics relative to weak baselines
  - but the emitted phoneme sequences still collapse heavily, often toward `SIL` or deletion-dominated behavior

## Interpretation

- the contrastive future-prediction objective is learning something with `S5`, but the current evidence suggests that much of the gain comes from shortcut-friendly temporal continuity and shard-local context rather than robust phoneme content
- low rank by itself is not necessarily bad for motor cortex data, but this particular low-rank structure currently looks more nuisance-aligned than speech-aligned
- high SSL top-1 should not be treated as meaningful progress unless it is accompanied by:
  - healthier geometry
  - weaker shard-local clustering
  - and better held-out phoneme transfer

## Current Conclusion

- the earlier direct future-prediction result remains negative
- the newer contrastive future-prediction variant is more promising than direct regression, but it still does not yet provide convincing evidence of a good phoneme-decoding basis
- future prediction with `S5` should remain in scope, but only with stronger anti-shortcut controls

## Next Changes To Prioritize

- use harder negative structure, especially same-session retrieval rather than unrestricted global negatives
- prefer longer horizons over very short local ones
- keep checking transfer with the cheap held-out phoneme probe rather than trusting SSL top-1 alone
