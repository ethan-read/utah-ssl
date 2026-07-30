# POSSM Stage-1 pooled-data discrepancy

## Question

The Brain2Text24-only Stage-1 run from `s6_possm_maskedreconstruction.ipynb`
showed a conventional reconstruction curve: train and validation MSE declined
smoothly, with validation often lower than training. The pooled Brain2Text24 +
Brain2Text25 run from `s14_possm_pooled_pretraining.ipynb` was much noisier:
validation MSE reached its minimum near step 2,000 and then increased, while
training MSE continued declining toward a plateau near 0.4 at step 12,000.

This note records confirmed differences that may explain that behavior. It is
intended as a handoff for designing the next diagnostic or corrected
experiment.

## Main conclusion

The Stage-1 model and optimizer configurations in s6 and s14 are effectively
the same. The runs differ substantially in their data domains, source-split
policies, signal representations, and validation composition.

The absolute MSE values are not directly comparable. Brain2Text24 and
Brain2Text25 are nominally session-z-scored before entering the model, but
they reach that normalization step through very different storage and
smoothing paths.

## Stage-1 settings that match

Both notebooks currently use:

- seed: `7`
- feature mode: `tx_only`
- input width: 128 area-6v TX channels
- boundary key: `session`
- segment length: 100 bins
- session-level feature-wise normalization
- model dimension: `64`
- latent count: `4`
- feed-forward hidden size: `512`
- dropout: `0.15`
- batch size: `32`
- steps: `12,000`
- learning rate: `3e-4`
- weight decay: `1e-3`
- validation every 50 steps
- validation batches per point: `2`
- dataset-weight exponent: `0.25`
- examples grouped per selected shard: `8`
- temporal backbone: one-layer unidirectional GRU
- objective: plain MSE
- masking: disabled
- linear reconstruction head

The larger loss and early pooled validation minimum are therefore not
explained by an intentional architecture, optimizer, masking, or step-budget
change.

## Selected-data sizes

Numbers below were measured from the local canonical raw cache at
`/Users/home/thesis/data/cache_v1`, using the source splits selected by s14.
One bin is treated as 20 ms (50 Hz).

| Quantity | B2T24 `competition_train` | B2T25 `train+val` | B2T25 / B2T24 |
|---|---:|---:|---:|
| Examples | 8,800 | 9,498 | 1.08x |
| Sessions | 24 | 45 | 1.88x |
| Neural bins | 2,777,568 | 8,376,668 | 3.02x |
| Approximate duration | 15.4 h | 46.5 h | 3.02x |
| Possible 100-bin windows | 1,906,369 | 7,436,366 | 3.90x |
| Mean example length | 316 bins | 882 bins | 2.79x |

The pooled selected corpus therefore contains approximately four times the
neural duration and 4.9 times the possible 100-bin windows of B2T24
`competition_train` alone.

This increase does not produce four times as many optimizer updates: both runs
use 12,000 steps with batch size 32.

With `dataset_weight_alpha=0.25`, the current pooled sampler selects
approximately:

- 41.1% Brain2Text24 windows
- 58.9% Brain2Text25 windows

Thus the pooled run gives B2T24 only about 41% as many sampled updates as a
12,000-step B2T24-only run. Dataset size alone should reduce corpus coverage,
but it does not by itself explain an early optimization plateau or rising
validation loss.

## Source-split difference

s14 explicitly uses:

- B2T24: `competition_train`
- B2T25: `train` and `val`
- B2T24 `competition_test`: excluded
- B2T25 `test`: excluded

s6 currently has no Brain2Text24 source-split filter. On the local canonical
manifest, that means the eligible B2T24 rows include:

- `competition_train`: 8,800 rows
- `competition_test`: 880 rows
- `none`: 6,408 rows
- total: 16,088 rows

Therefore, the historical s6 Stage-1 pool was not equivalent to the B2T24
portion of s14. It may have included all three source-split categories.

s6 also excludes only `brain2text25` by dataset name. If the smoothed cache
used by the original Colab run contained other dataset directories, those
would also have been eligible. The actual Stage-1 checkpoint configuration,
dataset counts, and cache inventory should be inspected to determine whether
this occurred in the completed baseline run.

## Subjects and channel identity

The selected datasets come from different subjects:

- Brain2Text24: `t12`
- Brain2Text25: `t15`

The current Stage-1 POSSM encoder uses one shared learned embedding for each
channel index. Although the sampler supplies session identifiers, the
`POSSMEncoder` currently ignores them and has no subject embedding or
session-specific input adapter.

Consequently, channel index 0 from T12 and channel index 0 from T15 are treated
as the same unit identity even though electrode placement and neural response
properties differ. A single shared reconstruction mapping must absorb that
cross-subject discrepancy implicitly.

More data does not necessarily lower loss when the added data come from
another subject and the model lacks an explicit domain-alignment mechanism.
The shared model can encounter conflicting channel mappings, reach a higher
approximation-error floor, and specialize to training sessions while
generalization to held-out sessions worsens.

## Normalization

Both workflows use per-session, per-channel z-scoring:

```text
z[t, c] = (x[t, c] - session_mean[c]) / max(session_std[c], 0.1)
```

Values are clipped to `[-20, 20]`. Session keys include the dataset name, so
Brain2Text24 and Brain2Text25 statistics do not collide.

The statistics are computed from the selected Stage-1 source splits. In s14,
the pooled statistics therefore cover B2T24 `competition_train` and B2T25
`train+val`.

Session z-scoring equalizes first- and second-order marginal scale per channel.
It does not make the datasets equivalent in:

- sparsity
- value distribution and kurtosis
- temporal autocorrelation
- cross-channel covariance
- effective dimensionality
- subject/electrode correspondence
- predictability or reconstruction difficulty

Brain2Text25 appears to contain continuous values that already resemble
z-scored threshold-crossing features before the shared Stage-1 normalization.
It is subsequently smoothed and session-z-scored again. This is not
necessarily incorrect, but it differs from the Brain2Text24 path.

## Smoothing and dtype asymmetry

The pre-smoothed cache builder currently performs smoothing in float32 and
then casts the result back to the source dtype:

```python
smoothed = _apply_gaussian_smoothing(...)
return smoothed.cpu().numpy().astype(array.dtype, copy=False)
```

Relevant source dtypes in the local canonical raw cache are:

- Brain2Text24 TX: `uint8`
- Brain2Text25 TX: `float32`

This creates materially different "smoothed" targets:

- B2T24 smoothing is cast back to integer `uint8`, truncating fractional
  Gaussian-smoothed values.
- B2T25 smoothing remains continuous floating point.
- The cache used by the completed pooled run stored B2T25 TX as `float16`;
  measured FP16 error was small compared with the preceding representation
  difference. The current preparation workflow preserves source dtypes and
  only selects area-6v columns `[0, 128)`.

A representative in-memory application of the current sigma-2 smoothing path
produced:

| Property after smoothing/storage | B2T24 | B2T25 |
|---|---:|---:|
| Stored representation | `uint8` | continuous float |
| Exact-zero fraction | 96.5% | approximately 0% |
| Adjacent values exactly equal | 98.7% | 9.2% |
| Mean absolute adjacent change after channel standardization | 0.065 | 0.180 |

These figures were measured from representative local raw shards by applying
the current smoothing implementation in memory. The corresponding Drive
smoothed shards should be audited directly before treating the exact values as
the definitive artifact statistics.

The B2T24 target is therefore extremely sparse and temporally constant after
the current smoothing/storage path. It is likely much easier to reconstruct
than the continuous B2T25 target. Session z-scoring does not undo this
quantization or restore discarded fractional values.

This is a strong explanation for s6 reaching MSE near 0.1-0.2 while the pooled
run remains much higher. The low s6 reconstruction loss may partly measure an
easy quantized target rather than a better representation.

## SBP comparison

SBP has a different profile from TX and does not show the same B2T24 integer
quantization problem. The local raw SBP arrays inspected for both datasets are
continuous `float32` values. B2T24 SBP is stored in a large positive power
scale, while B2T25 SBP already resembles session-normalized data:

| Property | B2T24 SBP | B2T25 SBP |
|---|---:|---:|
| Representative raw range | 59 to 69,596 | -2.84 to 10.0 |
| Raw mean across channels | 985 | -0.0035 |
| Raw mean channel standard deviation | 790 | 0.998 |
| Raw storage dtype | `float32` | `float32` |

These raw scales are not directly comparable, but session-wise feature
normalization is intended to remove that difference. Both SBP signals are
essentially dense and continuous: after sigma-2 smoothing, the exact-zero
fraction was effectively zero for both datasets, and exact adjacent-value
repetition was negligible.

A representative local audit of three shards per dataset gave the following
post-smoothing temporal comparison:

| Property after sigma-2 smoothing | B2T24 SBP | B2T25 SBP |
|---|---:|---:|
| Exact-zero fraction | 0% | approximately 0% |
| Adjacent values exactly equal | `1.1e-6` | `1.8e-7` |
| Mean absolute adjacent change after channel standardization | 0.113 | 0.145 |

B2T25 SBP therefore changes somewhat faster between adjacent bins, about
`1.28x` the standardized adjacent variation of B2T24, but the difference is
modest compared with the TX temporal/quantization mismatch. SBP appears much
more compatible across the two datasets after normalization.

This audit used the local canonical raw cache at
`/Users/home/thesis/data/cache_v1` and the local pre-smoothed B2T24 cache at
`/Users/home/thesis/data/cache_v1_smoothed_sigma2p0`. The local versioned
Brain2Text25 smoothed POSSM cache was not present, so B2T25 sigma-2 values were
computed in memory with the same smoothing implementation rather than read
from a stored smoothed artifact. The measurements are representative, not a
full-corpus SBP audit.

The completed s6 run and the earlier pooled s14 run audited in this document
used `tx_only`; the plotted loss mismatch therefore reflects TX. The current
s14 recipe instead uses 128-channel area-6v SBP for both years. Treat that as
a fresh experiment rather than as a resume of the earlier TX run.

## Validation construction and noise

The Stage-1 train/validation division is an internal deterministic
session-disjoint split for each dataset. B2T25 source `val` is not used
directly as the plotted validation partition. B2T25 `train+val` rows are first
pooled, then sessions are divided internally into Stage-1 train and
validation groups.

The plotted s14 validation MSE is an aggregate mixture of held-out B2T24 and
B2T25 sessions. It cannot show whether:

- B2T24 validation continues improving while B2T25 worsens;
- both datasets worsen;
- one dataset dominates the aggregate due to a higher MSE scale.

Only two validation batches (64 sampled windows) are evaluated at each point.
With two datasets of different reconstruction difficulty, random batch
composition and session selection make this estimate noisy. The s6 validation
estimate uses the same number of batches but is more homogeneous.

Training records also report one sampled mixed batch at each logged step.
Variation in B2T24/B2T25 composition and session difficulty contributes to the
noisier s14 training curve.

## Likely interpretation of the plateau

The pooled train MSE near 0.4 is probably an average of an easier B2T24 loss
and a harder B2T25 loss. For illustration, if the B2T24 component were around
0.1-0.2, the observed sampler weights would imply a B2T25 training component
roughly around 0.54-0.61.

The early validation minimum and later train/validation divergence are
consistent with one or more of:

1. unseen-session generalization failure concentrated in B2T25;
2. cross-subject interference from sharing channel embeddings across T12/T15;
3. a higher irreducible reconstruction floor for continuous B2T25 targets;
4. aggregate checkpoint selection being dominated by the higher-scale B2T25
   loss;
5. noisy two-batch validation estimates;
6. reduced B2T24 update exposure in the fixed 12,000-step pooled budget.

Simply extending Stage 1 is unlikely to reverse a validation curve that has
already risen for 10,000 steps. More steps may reduce training MSE while
increasing subject/session specialization.

## Checkpoint handoff

s14 normally hands `checkpoint_best.pt` to Stage 2 rather than the final
12,000-step checkpoint. "Best" currently means minimum aggregate pooled
validation MSE. If B2T25 dominates that metric, it may not select the best
checkpoint for the downstream B2T24/T12 decoding task.

Before starting the definitive Stage-2 comparison, evaluate several Stage-1
checkpoints separately on:

- held-out B2T24 sessions
- held-out B2T25 sessions
- substantially more than two batches per dataset

Candidate checkpoints include approximately steps 2k, 4k, 8k, and 12k. Since
the downstream task is B2T24, B2T24 validation behavior should be reported and
considered explicitly during checkpoint selection. A short matched downstream
probe is another possible selection criterion, provided it does not use the
held-out B2T24 test labels for model selection.

## Diagnostics needed

The next agent should consider implementing:

1. Per-dataset train and validation MSE during Stage 1.
2. Fixed, larger validation panels rather than two newly sampled batches.
3. Per-dataset zero-predictor and previous-bin-predictor MSE after the exact
   training normalization.
4. Histograms, sparsity, kurtosis, lag autocorrelation, cross-channel
   covariance, and effective rank after smoothing and normalization.
5. Direct audit of the Drive raw and smoothed shard dtypes and value
   distributions.
6. Inspection of the completed s6 and s14 checkpoint payloads:
   configuration, cache signature, dataset counts, best step, and histories.
7. Reconstruction evaluation of the same checkpoint separately on B2T24 and
   B2T25.
8. A subject-aware encoder variant, such as subject embeddings or
   subject/session-specific channel adapters.
9. A float-preserving smoothing cache for both datasets.

## Experimental-design cautions

- Rebuilding B2T24 smoothing in float32 would change the baseline target and
  require a fresh B2T24-only baseline for a controlled comparison.
- The existing pooled run is still informative as a test of adding the
  currently available B2T25 representation, but it does not isolate dataset
  quantity from subject and preprocessing differences.
- Do not compare absolute s6 and s14 MSE as though both measured the same
  target distribution.
- Do not choose the Stage-1 checkpoint using B2T24 competition-test
  reconstruction or labels.
- Keep Brain2Text24 `competition_test` and Brain2Text25 `test` out of any new
  pretraining or checkpoint-selection workflow intended to preserve the
  current held-out evaluation policy.

## Relevant files

- `analysis/reference/possm/notebooks/s6_possm_maskedreconstruction.ipynb`
- `analysis/reference/possm/notebooks/s14_possm_pooled_pretraining.ipynb`
- `analysis/reference/possm/notebooks/s15_possm_pooled_cache_preparation.ipynb`
- `analysis/reference/possm/possm_ssl/model.py`
- `analysis/reference/possm/possm_ssl/training.py`
- `analysis/reference/possm/possm_ssl/stage1_objectives.py`
- `analysis/active/ssl_experiments/masked_ssl/cache.py`
- `analysis/active/ssl_experiments/ssl_core/scripts/build_smoothed_cache.py`
- `analysis/active/ssl_experiments/ssl_core/scripts/prepare_possm_pooled_cache.py`
- `analysis/reference/possm/EXPERIMENT_NOTES.md`
- `docs/notes/cache_and_stats_inventory.md`
- `docs/notes/archive/possm_reproduction_results.md`
