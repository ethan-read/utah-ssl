# RunPod Workflow

This directory documents conventions shared by RunPod experiments. Individual
launch scripts and environment files live with their experiment branch; the
current supervised S5 launchers are under
`experiments/supervised_baselines/launchers/runpod/`.

Run from the repository root, mount or copy the canonical cache outside the
repository, and write recoverable outputs to persistent storage. Do not embed
machine-specific SSH credentials in tracked files.

RunPod storage may stage outputs during execution. Before a completed run is
entered into a canonical result report, copy its required evidence bundle to
Google Drive using the shared
[run artifact layout](../../docs/run_artifact_layout.md). Drive is the permanent
source of record; existing RunPod artifacts retain their current paths.
