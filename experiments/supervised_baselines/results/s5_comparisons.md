# Supervised S5 Comparisons

The supervised S5 decoder uses the Willett-derived preprocessing and CTC recipe
with temporal patches before the sequence model.

| Signal | Steps | Best validation PER | Final validation PER |
|---|---:|---:|---:|
| area-6v TX | 12,000 | `0.33637` | `0.33732` |
| area-6v TX+SBP | 60,000 | `0.25591` | `0.25736` |

The TX+SBP run has the lowest observed supervised validation PER currently
recorded in the repository. It used run name
`willett_s5_tx_sbp_seed7_60k`; its artifacts remain under the established Drive
output hierarchy. These within-run minima are not held-out test estimates.
