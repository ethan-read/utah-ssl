# Contrastive SSL Note

This note records the current status of the `S5` contrastive self-supervised learning experiments.

## Goal

The current workflow is:

1. Pretrain a causal `S5` encoder with a contrastive SSL objective (`future_infonce`) on the non-`Brain2Text25` corpus.
2. Evaluate the pretrained encoder on held-out `Brain2Text25` with a cheap downstream phoneme `CTC` probe.
3. Compare the pretrained encoder against simple baselines to determine whether SSL is learning a useful representation for later causal decoding.

## Current Downstream Setup

- downstream task: held-out-session phoneme decoding
- target session used so far: `t15.2025.04.13`
- target-session supervised split:
  - train examples: `69`
  - val examples: `25`
- current cheap downstream test:
  - freeze the encoder
  - train only a linear phoneme `CTC` head
  - report `val_ctc_bpphone` and phoneme error rate (`PER`)

## Results So Far

### SSL-Pretrained Frozen Linear Probe

- model: `ssl_checkpoint_linear_probe`
- encoder: pretrained with contrastive SSL, then frozen
- trainable weights: linear `CTC` head only
- validation `CTC` loss: `41.899433` bits per phoneme
- validation phoneme error rate: `0.776` (`77.6%`)

### Random-Init Full Fine-Tune

- model: `random_init_finetune`
- encoder: random initialization
- trainable weights: encoder plus `CTC` head
- validation `CTC` loss: `8.088148` bits per phoneme
- validation phoneme error rate: `0.867332` (`86.7%`)
- note: this is not the fair baseline for the frozen SSL linear probe, because this run trained the full encoder rather than only the head

### Majority-Phoneme Baseline

- most common validation phoneme on `t15.2025.04.13`: `SIL`
- `SIL` incidence on validation targets: `159 / 701 = 0.2268` (`22.68%`)
- rough token-error baseline from always predicting the majority phoneme: about `0.773` (`77.3%`)
- note: this is not exactly the same as `PER`, but it is a useful sanity-check baseline

## Interpretation

- The SSL-pretrained frozen probe is much better than the older `CTC` loss observed in the earlier 200-step probe comparison, but its `PER` is still close to the majority-phoneme sanity baseline.
- The random-init full fine-tune achieved a much lower `CTC` loss than the frozen SSL probe, but its `PER` remained poor.
- The key unresolved question is whether the SSL model is learning a genuinely useful phoneme representation rather than collapsing toward frequent phonemes or otherwise exploiting the `CTC` objective.

## Next Check

The fair comparison for the current claim is:

- `ssl_checkpoint_linear_probe`: pretrained `S5`, frozen encoder, train head only
- `random_init_linear_probe`: random `S5`, frozen encoder, train the same head only

That frozen-random linear-probe baseline is the next result needed to support the claim that contrastive SSL learned something useful.
