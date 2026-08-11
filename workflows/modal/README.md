# Modal Workflow

This directory contains only infrastructure shared across experiment branches:

- `create_utah_ssl_volumes.py`: create the persistent cache and output volumes.
- `extract_volume_archive.py`: extract uploaded cache archives into a volume.

Experiment-specific Modal launchers live under the owning branch's
`launchers/modal/` directory. Existing volume names and in-volume cache/output
layouts remain unchanged.

Modal volumes may stage new experiment outputs during execution. Before a
completed run is entered into a canonical result report, copy its required
evidence bundle to Google Drive using the shared
[run artifact layout](../../docs/run_artifact_layout.md). Drive, rather than the
Modal volume, is the permanent source of record. This does not require
migrating existing volume artifacts.
