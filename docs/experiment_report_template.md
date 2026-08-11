# Experiment Report Template

Use one report for a coherent research question or comparison. Related runs
belong in the same report; record every exact run as a separate row rather than
creating one report per run.

Copy this file into the relevant experiment's `results/` directory and replace
the prompts below. Historical reports do not need to be retrofitted unless they
are being revised.

Canonical result reports contain completed runs only. Keep planned, running,
failed, interrupted, and abandoned attempts in launch logs or working notes;
they are not result evidence. A completed run with poor performance is still a
completed result and should be included.

## Status

- **Report status:** complete or superseded
- **Last updated:** YYYY-MM-DD
- **Research question:** What comparison or claim does this report address?
- **Hypothesis:** What outcome was expected, and why?

## Comparison Design

State what changes between conditions and what is held fixed. Identify the
primary comparison before examining the results.

- **Datasets and splits:**
- **Signal and channels:** For B2T24/B2T25, note the area-6v and clipped-FP16
  SBP defaults, or explicitly justify a departure.
- **Preprocessing and normalization:**
- **Training budget and stopping rule:**
- **Primary metric:** Name the metric, data partition, aggregation, and
  checkpoint-selection rule. Distinguish a best observed validation value from
  a held-out test estimate.
- **Secondary metrics:**

## Runs

Use stable run IDs that also identify the corresponding artifact directory.
Include a run only when it completed its declared training and evaluation
procedure. If the procedure changed during execution, document the deviation
and decide whether it still constitutes a completed run before reporting it.

| Run ID | Model / initialization | Data and signal | Seed | Budget | Command or notebook | Drive artifact path | Primary result |
|---|---|---|---:|---|---|---|---:|
| `example_run` | | | | | | | |

Add configuration columns when they are central to the comparison. Put lengthy
configuration details in the run's external configuration artifact and link it
here; do not copy generated configuration files into the repository.

## Reproducibility

- **Git commit:**
- **Python and key dependency versions:**
- **Runtime and hardware:** Colab, Modal, RunPod, local, or other
- **Launcher or environment definition:**
- **Checkpoint and output roots:**

## Results

Report the primary comparison first. Include relevant controls, uncertainty
across seeds, and links to plots or machine-readable metrics where available.
Do not silently select the best run when multiple seeds or attempts exist.

## Interpretation

Separate observations from explanations. State what the evidence supports and
what remains speculative.

## Limitations and Confounds

Record incomplete controls, mismatched budgets, data leakage risks, or other
issues that constrain the conclusion. Do not turn execution failures into
scientific findings.

## Decision

State the practical consequence: continue, revise, deprioritize, or run a
specific missing control.

## Evidence

Every completed run must retain and link:

- the exact configuration;
- the progress log;
- the best checkpoint selected by the declared validation rule;
- the final checkpoint;
- machine-readable metrics; and
- the plots used to summarize or interpret the run.

Store new runs using the shared
[run artifact layout](run_artifact_layout.md).

If the best and final checkpoints are the same file, record that explicitly.
The Markdown report is the canonical narrative summary. A notebook may launch
or inspect a run, but it is not a required run artifact and must not be the only
place where configuration or results are recorded. Configurations, logs,
metrics files, checkpoints, plots, and other generated artifacts remain in
their documented Google Drive locations rather than being committed to the
repository. Record Drive-relative paths; public sharing links are not required.
