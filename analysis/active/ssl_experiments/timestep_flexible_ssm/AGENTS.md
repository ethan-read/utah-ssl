# Objective

This folder is for active supervised `S5`-style neural speech decoding
experiments that test timestep flexibility on Brain2Text24 phoneme-labeled
data.

# Current Research Direction

The immediate goal is to start from the Willett-style supervised reconstruction
recipe and adapt it into a timestep-flexible `SSM` decoder that is trained on
canonical Brain2Text24 `20 ms` bins, then evaluated under multiple effective
bin widths.

The first intended comparison path is:

- dataset: `brain2text24`
- labels: canonical phoneme `CTC` targets
- base training resolution: `20 ms`
- evaluation resolutions:
  - native `20 ms`
  - rebinned `40 ms`
  - rebinned `60 ms`
- model family: `S5`-based supervised decoder
- reference recipe: `analysis/active/ssl_experiments/s7_willett_reconstruction.ipynb`
  and `analysis/active/ssl_experiments/willett_reconstruction`

# Working Assumptions

Use the Willett-style supervised pipeline as the anchor for:

- split policy
- normalization scope
- online smoothing order
- patching conventions
- `CTC` evaluation and reporting

Treat timestep flexibility as the main experimental variable, not an excuse to
silently change the rest of the recipe.

Training should remain anchored on `20 ms` canonical cache data unless there is
a clearly documented reason to do otherwise.

Rebinned `40 ms` and `60 ms` variants should be constructed explicitly and
tracked as derived views of the same validation data, with enough provenance to
tell:

- how bins were merged
- whether `TX` and `SBP` were averaged
- how phoneme targets were transferred or aligned
- whether temporal patching changed with bin width

# Open Design Questions

The first implementation should keep these questions visible rather than
burying them in notebook state:

- whether the model should consume raw rebinned frames directly or keep a
  matched real-time patch duration and hop across bin widths
- whether timestep flexibility should be implemented through explicit
  timestep-conditioned modules, shared continuous-time parameterization, or a
  simpler matched-baseline wrapper first
- whether validation should compare:
  - one model trained at `20 ms` and evaluated at multiple bin widths
  - versus separate models retrained per bin width

Default to the simpler, auditable comparison first:

- train on `20 ms`
- evaluate on `20 ms` and rebinned `40 ms`
- keep patch duration and hop fixed in milliseconds

# Reference Baselines

The main baseline remains the supervised Willett-style decoder in
`analysis/active/ssl_experiments/willett_reconstruction`.

Generic `SSL` packages such as `analysis/active/ssl_experiments/ssm_ssl` are
useful references for reusable `S5` modules, but this folder is not for
generic self-supervised modeling.

# Cleanup Direction

Prefer a small, scriptable experiment package over notebook-only state.

Keep:

- data transforms that define `20 ms -> 40 ms -> 60 ms` evaluation views
- shared evaluation utilities
- tests that protect rebinning semantics and target alignment

Avoid:

- one-off notebooks that duplicate the supervised baseline without narrowing a
  specific timestep-flexibility question
- ambiguous run names that do not encode train bin size, eval bin size, and
  patching mode
