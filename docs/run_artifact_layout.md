# Run Artifact Layout

Google Drive is the canonical permanent artifact store. Use this layout for new
runs beneath `/content/drive/MyDrive/utah_ssl/outputs` in Colab or the equivalent
synced local path `/Users/home/My Drive/utah_ssl/outputs`. Do not move or rename
existing artifacts to match it; historical reports should continue recording
their established locations.

Under the persistent output root, organize artifacts as:

```text
outputs/
└── <experiment_branch>/
    └── <comparison_name>/
        └── <run_id>/
            ├── config.yaml
            ├── progress.jsonl
            ├── metrics.json
            ├── checkpoint_best.pt
            ├── checkpoint_final.pt
            ├── plots/
            │   ├── training_curves.png
            │   └── evaluation_summary.png
            └── checkpoints/          # optional recovery checkpoints
```

`config.json` may replace `config.yaml` when JSON is already native to the
launcher. The meaning of the file must remain the same.

## Directory Identities

- `<experiment_branch>` matches the owning source branch, such as
  `supervised_baselines`, `bit_style`, `possm_style`, or `manifolds`.
- `<comparison_name>` names the research question or controlled comparison and
  normally corresponds to one result report.
- `<run_id>` identifies one exact configuration and seed. It should be stable,
  human-readable, unique within the comparison, and must not be reused for a
  different configuration.

A useful run ID includes the model or condition and seed, for example
`s5_pretrained_seed7` and `s5_random_init_seed7`. Add a timestamp or short
configuration identifier only when needed to prevent collisions.

## Required Completed-Run Artifacts

- `config.yaml` or `config.json` is the complete resolved configuration,
  including defaults—not merely the command-line overrides.
- `progress.jsonl` is the chronological, append-only training and evaluation
  record.
- `metrics.json` records the declared selection metric and partition, best
  step and metrics, and final step and metrics in machine-readable form.
- `checkpoint_best.pt` is selected using the validation rule declared before
  interpreting the result.
- `checkpoint_final.pt` is the state at normal completion of the run.
- `plots/` contains the figures used in the result report. Plot names may vary
  when `training_curves.png` and `evaluation_summary.png` are not appropriate.

If the best and final checkpoint are the same state, keep the filenames'
meanings unambiguous and record the equality in `metrics.json` and the result
report. Periodic files under `checkpoints/` are optional recovery state and are
not part of the permanent evidence bundle.

## Reporting and Storage

Only completed runs appear in canonical result reports. Failed, interrupted,
planned, and abandoned attempts may retain partial directories for diagnosis
or recovery, but those directories are not scientific evidence.

The repository stores the Markdown result summary, not copies of run artifacts.
Configurations, logs, metrics files, checkpoints, plots, and other generated
details remain under the Drive output root. Reports record Drive-relative paths;
public download URLs are not required. A report may reproduce a small metric
table needed to state its conclusion, but the machine-readable source remains
in Drive. A notebook may launch or inspect the run, but it is not a required
artifact and must not be the only record of configuration or results.

Colab writes directly to Drive. Modal, RunPod, and local execution may stage
artifacts elsewhere while a run is active, but the complete required evidence
bundle must be copied into the corresponding Drive directory before the run is
added to a canonical result report. Remote storage is execution or transfer
storage, not the permanent source of record.
