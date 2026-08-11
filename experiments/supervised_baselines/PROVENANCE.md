# Provenance

## Current Understanding

The GRU implementation in this experiment was produced through an LLM-assisted
Python port and adaptation of Stanford/Willett speech-decoder code. Based on the
associated checkpoint-conversion and evaluation work, the likely source was a
released TensorFlow implementation, but the exact source has not been verified.

This is not an official Stanford/Willett implementation. It should be treated
as upstream-derived code, not as an independently implemented decoder.

## Unresolved Details

The following must be established before this implementation is included in a
public release:

- the exact upstream repository, files, version, and commit;
- which portions were ported and which were subsequently adapted locally;
- the applicable license, copyright terms, and required notices; and
- whether redistribution is permitted or the affected portions need to be
  replaced with an independent implementation.

This note is an interim provenance disclosure, not legal or licensing
clearance. It belongs with the experiment and should move with it if the
experiment is relocated, including a future move to
`experiments/supervised_baselines/willett_gru/`.
