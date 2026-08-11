# Supervised baseline design

The GRU, S5, and S4D experiments share one data, temporal-patching, CTC, and
evaluation implementation so that comparisons isolate the sequence backbone.
Model-specific choices and unresolved recipe differences are recorded with the
corresponding reports under `../results/`.

The Willett GRU provenance constraint is documented separately in
[`../PROVENANCE.md`](../PROVENANCE.md).
