# Colab Workflow

Colab is the default environment for interactive development and most
experiments. Runnable notebooks live with their experiment branches rather
than in this directory.

For the supported Python version, lightweight local dependencies, and standard
test commands, see [the setup guide](../../docs/setup.md).

The standard workflow is:

1. Open the relevant branch notebook.
2. Mount Google Drive and clone or update this repository under `/content`.
3. Run from the repository root so `utah_ssl.*` and `experiments.*` imports
   resolve without branch-specific `PYTHONPATH` changes.
4. Read caches and reusable statistics from
   `/content/drive/MyDrive/utah_ssl/data`.
5. Write checkpoints and logs beneath
   `/content/drive/MyDrive/utah_ssl/outputs`.
6. Resume interrupted runs from Drive-backed checkpoints rather than notebook
   state.
7. For every completed run, retain its exact configuration, progress log, best
   and final checkpoints, machine-readable metrics, and result plots. The
   notebook itself is an execution interface, not a required run artifact. Use
   the shared [run artifact layout](../../docs/run_artifact_layout.md) for new
   runs.

Google Drive is the canonical permanent store for run artifacts, so no
additional public artifact upload is required.

The Colab CLI may be used to launch or synchronize notebooks when available,
but notebooks remain the canonical executable entry points. CLI commands should
be verified against the installed Colab CLI version before they are added to a
runbook.
