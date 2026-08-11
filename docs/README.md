# Documentation

- [Research status](research_status.md) summarizes the current cross-branch
  interpretation.
- [Setup and verification](setup.md) documents the supported Colab-first and
  lightweight local workflows.
- [Data documentation](data/README.md) records cache, signal, and normalization
  contracts.
- [Paper notes](paper_notes/README.md) contain source-specific architecture
  notes.
- Each experiment owns its reasoning, run instructions, and canonical results
  under `experiments/<branch>/`.
- [Experiment report template](experiment_report_template.md) defines the
  comparison-oriented format for new result reports and run records.
- [Run artifact layout](run_artifact_layout.md) defines the required evidence
  bundle and storage convention for new runs.

Detailed experiment results should not be added to this directory. Put them in
the relevant branch's `results/` directory and link them from the branch result
index.
