---
name: launch-decoding-run
description: Preflight, launch, resume, and monitor reproducible Utah-array decoding experiments across Colab, Modal, RunPod, or local execution. Use when starting or resuming a training run, preparing an exact launch command, checking whether a remote run is safe to start, monitoring an active run, or promoting a completed run's artifacts to Google Drive. Do not use for experiment ideation, code-only implementation, or interpretation of finished scientific results.
---

# Launch Decoding Run

Launch only the experiment the user has authorized. Preserve the declared scientific comparison, exact recovery semantics, and permanent evidence bundle across execution environments.

## Establish the run contract

1. Work from the repository root and read `AGENTS.md`.
2. Read these canonical files:
   - `docs/data/cache_and_stats_inventory.md`
   - `docs/data/signal_and_dataset_contracts.md`
   - `docs/run_artifact_layout.md`
   - the owning experiment's `README.md`, design note, notebook, and launcher documentation
   - the relevant file under `workflows/`
3. Treat Colab as the default. Use Modal, RunPod, or local execution only when the request or experiment's maintained launcher points there.
4. State the complete run contract before launch:
   - experiment branch, comparison name, run ID, and seed;
   - `DatasetPlan`, source splits, and `SignalSpec`;
   - cache roots, statistics artifacts, preprocessing, and normalization;
   - architecture, initialization, precision, training budget, and stopping rule;
   - primary metric, evaluation partition, and checkpoint-selection rule;
   - runtime, hardware, launcher, and permanent Drive output path.
5. Ask for user input only when a missing choice would materially change that contract. Do not silently choose a scientifically different configuration.

For Brain-to-Text 2024 and 2025, require the first 128 area-6v channels and clipped-FP16 SBP caches unless the experiment explicitly studies a documented departure. Use TX for broad auxiliary-dataset work when SBP is unavailable.

## Run the preflight

Perform and report these checks before consuming remote compute:

- Record `git rev-parse HEAD` and `git status --short`. Ensure the remote environment can access the intended commit. Do not commit or push unless the user authorized it.
- Run the relevant lightweight tests or launcher smoke checks from `docs/setup.md` when feasible.
- Inspect launcher help, configuration cells, and environment definitions rather than reconstructing commands from memory.
- Verify every selected dataset directory has its manifest, metadata, and shards.
- Verify cache identity, modality, channel count, source splits, smoothing state, and storage representation.
- Verify each statistics payload has its provenance sidecar and matches the full dataset plan, signal, cache identity, normalization scope, and split policy.
- Reject double smoothing and normalization fitted using evaluation examples.
- Choose a stable, unique run ID. Never reuse an existing run directory for a different resolved configuration.
- Resolve the permanent path as `outputs/<experiment_branch>/<comparison_name>/<run_id>/` under the Drive root. Preserve established historical paths instead of moving them.
- Confirm Python, key packages, accelerator support, available disk, persistent storage, and required credentials. Keep secrets in environment variables or platform secret stores; never write them into notebooks, commands saved in reports, or tracked files.

If any contract check fails, stop before launch and report the exact mismatch plus the least invasive correction.

## Handle fresh and resumed runs

For a fresh run, confirm that the target directory is absent or intentionally empty and save the complete resolved configuration before training begins.

For a resumed run:

1. Read the checkpoint contract rather than inferring it from the filename.
2. Match dataset plan, signal, cache identity, normalization, architecture, optimizer compatibility, and existing budget.
3. Resume Stage 1 or Stage 2 only from the recovery sources allowed by `docs/data/signal_and_dataset_contracts.md` and the owning experiment.
4. Never use `checkpoint_best.pt` as an exact-resume checkpoint when the contract identifies it only as an evaluation artifact.
5. Do not resume across changed shard topology, signal modality, architecture, or incompatible preprocessing. Start a new run ID instead.

## Launch and monitor

Run the maintained notebook or launcher from the repository root. Preserve the exact command or notebook configuration with the run artifacts.

During execution:

- Keep progress logs append-only.
- Check that steps, losses, validation metrics, throughput, cache behavior, and hardware match the declared run.
- Distinguish ordinary metric noise from data, numerical, or recovery failures.
- Never change the configuration mid-run to rescue a result. If a correction changes the scientific contract, stop and create a new run.
- When monitoring was requested, continue until the declared terminal condition or an actionable failure; provide concise status updates during long work.
- Stop or release paid compute after artifacts are safely persisted when the launcher and user authorization include that action.

## Close the run

Classify the run before touching canonical result documentation:

- **Completed:** finished the declared training and evaluation procedure.
- **Interrupted/recoverable:** may resume, but is not result evidence.
- **Failed/invalid:** retain diagnostic artifacts if useful, but do not interpret the failure as a scientific result.

For a completed run, verify the required bundle from `docs/run_artifact_layout.md`: resolved config, append-only progress log, machine-readable metrics, best checkpoint, final checkpoint, and report figures. Copy the complete bundle from Modal, RunPod, or local staging storage into the canonical Drive directory before it is added to a result report.

Do not update a canonical result report merely because a job started or produced a promising intermediate metric. Use `$evaluate-decoding-result` after completion.

## Report the outcome

Return a compact launch record containing:

- run ID and scientific contract;
- Git commit and exact execution entry point;
- runtime/hardware and active artifact path;
- whether the run was launched, resumed, completed, interrupted, or blocked;
- any deviations or unresolved risks;
- the next operational action, if one remains.
