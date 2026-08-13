---
name: evaluate-decoding-result
description: Audit completed Utah-array decoding runs, compare conditions, and determine what scientific claims the evidence supports. Use when reviewing metrics or artifacts, comparing runs or seeds, deciding whether a result is conclusive or confounded, writing or updating an experiment result report, or deciding whether a finding belongs in docs/research_status.md. Do not use to launch training or to treat incomplete and failed runs as scientific evidence.
---

# Evaluate Decoding Result

Evaluate the evidence independently of the hoped-for outcome. Separate artifact validity, numerical observations, scientific interpretation, and the next decision.

## Gather canonical context

1. Read `AGENTS.md` and identify the experiment that owns the run.
2. Read:
   - `docs/experiment_report_template.md`
   - `docs/run_artifact_layout.md`
   - `docs/data/signal_and_dataset_contracts.md`
   - the branch's design notes and canonical results
   - `docs/research_status.md` only when assessing a cross-cutting conclusion
3. Inspect the Drive-backed run artifacts in addition to the repository when they are needed. Treat Drive as the permanent source of record.
4. Preserve a read-only review unless the user also asked to update documentation.

## Pass the evidence gate

Resolve the exact run ID and artifact directory. Require:

- complete resolved configuration, including defaults;
- chronological progress log;
- machine-readable metrics;
- best and final checkpoints;
- the plots used for interpretation;
- Git commit, environment, runtime, hardware, seed, and declared budget.

Confirm that the run completed its declared training and evaluation procedure. Reconcile the config, command or notebook, logs, checkpoint metadata, and metrics. Record any mid-run procedure change as a deviation.

Classify an incomplete, interrupted, crashed, corrupted, or contract-incompatible run as non-evidence. It may still support an engineering diagnosis, but it must not enter a canonical result report.

## Audit metric selection

Identify the primary metric, partition, aggregation, validation cadence, and checkpoint-selection rule declared for the comparison. Then verify:

- `checkpoint_best.pt` was selected by the declared rule;
- the reported value belongs to the named partition and checkpoint;
- best observed, final, validation-selected, and held-out values are not conflated;
- repeated validation checks or sampled validation batches are not presented as stable held-out estimates;
- all relevant seeds and completed attempts are visible rather than silently selecting the best one.

When a historical workflow repeatedly evaluated a nominal test split during training, describe it as the workflow's validation/model-selection split. Do not call its selected minimum an untouched test estimate.

## Audit comparability

Compare conditions across these dimensions:

- datasets, source splits, subjects, and exposure counts;
- signal modality, channels, cache representation, smoothing, and normalization;
- architecture, initialization, trainable parameters, and parameter count;
- training budget, stopping rule, validation cadence, and selection rule;
- seed, split realization, precision, augmentation, runtime, and hardware;
- evaluation code, vocabulary, decoding procedure, and metric aggregation.

Call an effect isolated only when the intended factor changes and material alternatives remain fixed. Otherwise name every important confound. A larger data pool, longer budget, different precision, or changed signal may yield a useful result without estimating the causal contribution of any one change.

## Quantify without overstating

- Report the primary comparison first.
- Give absolute differences; add relative differences only with an explicit denominator and direction.
- Summarize every completed seed. Report spread or uncertainty when the design supports it.
- Do not manufacture statistical confidence from a single seed, repeated checkpoints, or correlated trials.
- Prefer paired comparisons when seeds and splits are matched; otherwise state the mismatch.
- Treat small best-step differences cautiously when validation is noisy or repeatedly sampled.
- Check calibration diagnostics such as predicted/reference length, blank rate, and validation loss when they affect interpretation of PER.

Assign the conclusion one evidence level:

- **Supported:** a valid, sufficiently controlled comparison directly supports the claim.
- **Suggestive:** the observation is real but single-seed, noisy, or materially confounded.
- **Not established:** artifacts, controls, or comparability are insufficient.
- **Contradicted:** valid evidence directly opposes the claim.

## Write the scientific interpretation

Structure the evaluation as:

1. **Observation:** what the artifacts and metrics show.
2. **Interpretation:** the narrowest explanation supported by the comparison.
3. **Limitations and confounds:** what prevents a stronger claim.
4. **Decision:** continue, revise, deprioritize, or run a specific missing control.

Never infer that an architecture is scientifically poor from a failed launch, broken kernel, incompatible checkpoint, or shallow exploratory attempt.

## Update documentation only when requested

When asked to record the result:

1. Use one coherent comparison report under the owning branch's `results/` directory, following `docs/experiment_report_template.md`.
2. Include completed runs only. Keep failed, abandoned, running, and interrupted attempts in operational logs or working notes.
3. Store only the Markdown narrative and compact metric tables in Git. Link Drive-relative artifact paths; do not commit generated plots, checkpoints, logs, or configurations.
4. Update the branch results index if needed.
5. Update `docs/research_status.md` only for a durable, cross-experiment conclusion. Preserve uncertainty and do not promote a single exploratory observation into project status.
6. Run Markdown-link checks where available and `git diff --check` after editing.

Return the evidence level, primary numerical comparison, major confounds, documentation changes, and the single most informative next control.
