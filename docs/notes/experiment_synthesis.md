# Experiment Synthesis

This is the current canonical interpretation of the Utah SSL / speech-decoding
experiment logs. It is organized around research claims rather than around the
order the experiments were run.

Historical result ledgers live in [`archive/`](archive/). Those files are kept
as evidence, but they may contain superseded paths, stale hyperparameters, or
intermediate interpretations.

Recent S4D sweep values are taken from the Drive aggregate files under
`/content/drive/MyDrive/utah_ssl/outputs/willett_s4d_hyperparam_sweep/my_s4d_sweep/`.

## Current Headline

The strongest evidence so far is that the Willett-style supervised CTC recipe is
the reliable baseline to beat. Supervised sequence models train well when they
receive Willett-style temporal patches before the sequence backbone. The current
self-supervised lines have not yet produced a representation that matches this
supervised decoding strength.

## Claim 1: Supervised Willett-Style Decoding Works

The local supervised setup now has a credible comparison ladder over GRU, S5,
and S4D backbones:

| Backbone | Current best visible result | Interpretation |
|---|---:|---|
| GRU | latest visible S7 progress log: `PER=0.37485` at step `9400` | Useful reproduction baseline, but the notebook contains mixed historical outputs and the recipe still has known discrepancies. |
| S5 | best observed `PER=0.33637`, final `PER=0.33732` at `12000` steps | Strongest supervised result in these notes; S5 is not the limiting factor when paired with temporal patching. |
| S4D | initial 12k run best `PER=0.37526`; 8k sweep baseline best `PER=0.39058` | Trains and decodes, but looks slightly weaker than S5 on the runs so far. |

Important caveat: these are still mostly single-seed comparisons. The S4D
hyperparameter sweep was small and did not improve on the default S4D recipe:

| S4D variant | Steps | Best PER | Final PER | Notes |
|---|---:|---:|---:|---|
| baseline | `8000` | `0.39058` | `0.39128` | Best small-sweep result. |
| lower learning rate | `8000` | `0.45044` | `0.45131` | Too conservative / under-emissive. |
| lower dropout | `8000` | `0.40957` | `0.41006` | Worse than baseline. |
| wider | `8000` | `0.40994` | `0.41143` | Doubled parameters without helping. |

Current interpretation:

- `S5` is the strongest supervised SSM result so far.
- `S4D` is viable but not yet competitive with `S5` under the tested settings.
- The GRU reproduction is still useful as the recipe anchor, but it should be
  interpreted together with [`willett_reconstruction_replication.md`](willett_reconstruction_replication.md)
  because that note tracks remaining recipe discrepancies.

## Claim 2: POSSM Pretraining Helps, But Does Not Yet Close The Gap

The POSSM path has moved from collapsed or near-collapsed decoding into a real,
measurable decoding regime. The best current POSSM stage-2 result in the archived
ledger reached about `PER=0.467` at `12000` steps on area-6v `tx_only`.

The strongest controlled lesson is that reconstruction pretraining helps:

- pretrained POSSM stage-2 decoding reached roughly `PER=0.467`
- a matched random-init stage-2 baseline was much worse, around
  `PER=0.707-0.728`
- the random-init model stayed more blank-dominant and under-emissive

But POSSM still trails the supervised Willett-style baselines. The most likely
explanation is not that SSMs cannot decode this dataset. Supervised S5 performs
well when it receives temporal patches before the sequence model. The more
plausible issue is the POSSM stage-2 temporal interface and fine-tuning recipe:
POSSM emits one encoded `20 ms` frame at a time before the decoder, while the
Willett-style supervised models give the backbone a local temporal patch before
sequence modeling.

Current interpretation:

- POSSM-style reconstruction pretraining is useful for optimization.
- The stage-2 decoding interface is probably still underpowered or mismatched.
- A targeted POSSM follow-up should test temporal patching over POSSM encoder
  outputs before the S5/S4D/GRU decoder.

## Claim 3: Current SSL Objectives Have Not Yet Produced A Useful Phoneme Basis

Several SSL directions have been informative, but none currently provides
convincing evidence of a phoneme-discriminative representation:

- Direct S5 future prediction was negative; it did not clearly beat the
  normalized zero-prediction baseline.
- Contrastive future prediction can achieve high retrieval metrics, but the
  embeddings appear shortcut-prone: local shard/session structure and low-rank
  geometry dominate, and downstream phoneme probes remain weak.
- Causal S5 masked reconstruction is implemented and instrumented, but the
  early versions collapsed toward the normalized-mean predictor or showed only
  modest reconstruction gains without useful downstream transfer.
- POSSM reconstruction pretraining is the exception: it helps stage-2 CTC
  optimization, but it is still not enough to match supervised decoding.

Current interpretation:

- Do not use SSL retrieval accuracy or reconstruction loss alone as proof of
  progress.
- Downstream phoneme decoding and emission diagnostics remain the decisive tests.
- The current bottleneck is more likely objective/interface design than raw
  model capacity.

## Claim 4: Data And Recipe Fidelity Matter

Several non-model details have materially changed how runs should be interpreted:

- Active Brain2Text24 feature policy is area-6v only:
  - `tx_only`: first `128` area-6v TX features
  - `tx_sbp`: first `128` area-6v TX plus first `128` area-6v SBP features
- Stage-2 phoneme fine-tuning should use train-split global stats from
  `competition_train` and apply those same stats to validation.
- Willett-style online smoothing should happen after normalization and training
  augmentations for CTC fine-tuning.
- Reusable stats are data artifacts under `utah_ssl/data/stats`, not experiment
  outputs.

See [`cache_and_stats_inventory.md`](cache_and_stats_inventory.md) before
changing cache roots, stats paths, feature modes, or smoothing policy.

## Superseded Or Inactive Lines

The following notes have been archived because they are historical ledgers rather
than current source-of-truth documents:

- [`archive/possm_reproduction_results.md`](archive/possm_reproduction_results.md)
- [`archive/s5_masked_reconstruction_note.md`](archive/s5_masked_reconstruction_note.md)
- [`archive/s5_future_prediction_note.md`](archive/s5_future_prediction_note.md)
- [`archive/contrastive_ssl_note.md`](archive/contrastive_ssl_note.md)
- [`archive/channel_ablation_summary.md`](archive/channel_ablation_summary.md)

They should still be consulted when reconstructing how a conclusion was reached,
but new decisions should start from this synthesis and the active operational
notes.

## Current Priorities

1. Treat supervised Willett-style S5 as the strongest current baseline.
2. Keep S4D as a viable but secondary supervised SSM baseline unless seed sweeps
   or targeted tuning change the picture.
3. Improve POSSM stage-2 by testing a Willett-style temporal patch interface over
   POSSM encoder outputs.
4. Interpret SSL objectives through downstream phoneme transfer, not SSL loss or
   retrieval metrics alone.
5. Keep recipe-fidelity fixes visible in
   [`willett_reconstruction_replication.md`](willett_reconstruction_replication.md),
   especially validation normalization, learned initial states, and day-wise
   sampling.
