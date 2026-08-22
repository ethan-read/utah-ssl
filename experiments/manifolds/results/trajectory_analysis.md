# Neural Trajectory Analysis Status

The current workflow preserves utterance time order, aligns events using model
CTC timing, and tests within-phoneme trajectory repeatability and cross-session
robustness. It includes input-window controls and reconstructs covered 20 ms
bins for the strict robustness analysis.

The analysis remains exploratory. Decoder timing is not independent neural
timing, overlapping input windows mix temporal information, and a path in model
state space is not by itself evidence for an intrinsic biological dynamical
system.

## Raw-bin PCA

The completed raw 20 ms SBP analysis did not find a useful small global linear
PCA subspace in pooled instantaneous activity. This deprioritizes global PCA as
the representation for the next manifold analysis, but does not rule out
low-dimensional shared, temporal, condition-specific, or nonlinear structure.
Artifacts are stored at
`outputs/neural_trajectories/raw_20ms_sbp_pca_t12_chronological_v1` on Drive.

Leave-session-out trajectory controls remain to be fully interpreted.
