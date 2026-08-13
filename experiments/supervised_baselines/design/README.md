# Supervised baseline design

The GRU, S5, and S4D experiments share one data, temporal-patching, CTC, and
evaluation implementation so that comparisons isolate the sequence backbone.
Model-specific choices and unresolved recipe differences are recorded with the
corresponding reports under `../results/`.

The Willett GRU provenance constraint is documented separately in
[`../PROVENANCE.md`](../PROVENANCE.md).

## Shared Willett-style preprocessing

The decoder branches use the shared smoothing and training-augmentation code
in `utah_ssl.decoding_preprocessing`. It follows the repository's adapted
Willett-style pipeline: thresholded Gaussian smoothing, white-noise
augmentation, and per-example constant offsets. This is an adaptation of the
local Willett-derived path, not a universal data default or an official
Stanford implementation; exact upstream provenance remains unverified as
described in [`../PROVENANCE.md`](../PROVENANCE.md).

| Source element | Local implementation | Classification | Validation |
|---|---|---|---|
| Willett-style smoothing and augmentation recipe | `utah_ssl.decoding_preprocessing` | Adapted; exact upstream files unverified | Shared core, supervised, BIT, and POSSM tests |

Checkpoint architecture reconstruction is owned by
`experiments.supervised_baselines.checkpointing`. Repository-trained and
converted released checkpoints use the same model factory; explicit adapter
keys stored in a checkpoint take precedence over keys reconstructed from the
selected dataset problem. This centralization does not change the provenance
classification of converted Willett-derived checkpoints.
