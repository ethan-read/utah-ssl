# Supervised Baseline Results

New comparisons should use the shared
[experiment report template](../../../docs/experiment_report_template.md), with
related runs grouped in one report and exact runs recorded as rows. Existing
historical summaries are retained as written.

Unless a report says otherwise, headline values below are the lowest observed
PER on the run's validation split. They are not held-out test estimates.

| Experiment | Best observed validation PER | Canonical report |
|---|---:|---|
| Willett-derived GRU | `0.37485` | [GRU replication](willett_gru_replication.md) |
| Supervised S5, TX only | `0.33637` | [S5 comparisons](s5_comparisons.md) |
| Supervised S5, TX+SBP | `0.25591` | [S5 comparisons](s5_comparisons.md) |
| Supervised S4D | `0.37526` | [S4D comparison](s4d_comparison.md) |

These are mostly single-seed results and should not be presented as a
statistically resolved architecture ranking.
