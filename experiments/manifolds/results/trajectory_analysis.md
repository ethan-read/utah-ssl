# Neural Trajectory Analysis Status

The current workflow preserves utterance time order, aligns events using model
CTC timing, and tests within-phoneme trajectory repeatability and cross-session
robustness. It includes input-window controls and reconstructs covered 20 ms
bins for the strict robustness analysis.

The analysis remains exploratory. Decoder timing is not independent neural
timing, overlapping input windows mix temporal information, and a path in model
state space is not by itself evidence for an intrinsic biological dynamical
system. Canonical conclusions should be updated here after the raw-bin and
leave-session-out controls are fully interpreted.
