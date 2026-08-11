# Channel Ablation Summary

- Source notebook: not present in the repository at migration time.
- Method: rank electrodes by occlusion importance on a small train calibration subset, then keep the top `50%` and evaluate on the full validation split.

## Result

- Baseline aggregate PER: `0.10096`
- 50% retained aggregate PER: `0.24664`
- Absolute PER increase: `+0.14568`
- Validation sessions: `41`
- Validation trials: `1426`
- Paired Wilcoxon p-value: `9.09e-13`

## Interpretation

- This run does **not** support the claim that half of the electrodes can be removed with little loss in decoding accuracy.
- Under this occlusion-ranked mask, performance degrades substantially and consistently across sessions.

## Important Note

- A reporting bug sorted the retained and dropped electrode IDs before printing the "Top retained" and "Top dropped" lists.
- This did **not** affect the actual mask used for evaluation; it only affected how those lists were displayed.
