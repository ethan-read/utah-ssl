# Experiments

Each active folder represents a specific research idea and owns its notebooks,
launchers, design reasoning, tests, and results.

## Active

- [Supervised baselines](supervised_baselines/README.md)
- [BIT-style pretraining](bit_style/README.md)
- [POSSM-style pretraining](possm_style/README.md)
- [Manifold and trajectory analyses](manifolds/README.md)

New branches should be named for a concrete hypothesis or mechanism. Do not
create a general `transfer_learning` folder.

Use the [experiment report template](../docs/experiment_report_template.md) for
new result summaries. Group related runs by research question or comparison and
record each exact completed run as a row in the report. Keep failed,
interrupted, planned, and abandoned attempts out of canonical result reports.
Each reported run must retain its configuration, progress log, best and final
checkpoints, machine-readable metrics, and result plots; the execution notebook
is not part of the required evidence bundle. These generated artifacts stay in
persistent external storage; only their Markdown result summary belongs in the
repository.

Inactive work belongs under [archive](archive/README.md) and must not be
imported by active branches.
