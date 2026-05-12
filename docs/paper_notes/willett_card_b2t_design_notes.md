# Willett / Card Brain-to-Text Design Notes

Paper / code family:

- `A high-performance speech neuroprosthesis` (Willett et al., Nature 2023)
- `An accurate and rapidly calibrating speech neuroprosthesis` (Card et al., NEJM 2024)
- local code reference: `/Users/home/thesis/code/brain2text25`

## Scope Of These Notes

These are brief design notes for the Stanford Utah-array speech-decoding stack, aimed at remembering architecture and preprocessing choices that may be useful while training SSL models on larger Utah-array corpora.

They are not reproduction instructions.

## High-Level Takeaways

- the stack is still fundamentally `RNN + CTC + LM`, not a transformer encoder-decoder
- strong performance comes from a relatively plain causal neural decoder plus careful feature handling, day/session adaptation, and a strong external language model
- nonstationarity handling is treated as a first-class systems problem
- the model uses explicit session/day-specific input adaptation rather than forcing one shared model to absorb all drift
- temporal downsampling is done by patching / striding the neural sequence before the recurrent core

## Neural Input Representation

- Utah-array features are binned at `20 ms`
- the standard speech-decoding input is `TX + SBP`
- for `256` electrodes this becomes `512` input features per timestep
- in the competition README they explicitly suggest starting with:
  - `spikePow`
  - `tx1`
  - optionally restricting to the first `128` features for area `6v`

Why this matters for SSL:

- the Stanford stack does not rely on sophisticated tokenization
- it gets a lot of mileage out of simple dense per-bin features
- `TX` and `SBP` are treated as complementary, not as separate streams

## Normalization And Drift Handling

This is one of the most important design choices.

What is explicit in the local code and docs:

- the training code reads precomputed `input_features` from HDF5 directly
- the Python trainer does not implement the online normalization itself
- the competition data README recommends `blockIdx`-based blockwise z-scoring because mean drift can be severe

What is explicit in the papers:

- feature means are corrected to handle nonstationarity
- online decoding uses a `rolling z-scoring` stage

What is clearest from the patent / supplementary public description:

- rolling z-scoring operates on windows of roughly `1-10 min`
- a new window is warm-started from the previous window's statistics
- during the first `5-15` sentences of a new window, with `10` given as the example, normalization blends:
  - previous-window stats
  - current-window stats estimated from the first few sentences in the new window
- after that warmup phase, the previous-window stats are dropped
- standard deviation is updated analogously to the mean

The practical lesson is:

- this is not a naive per-bin causal z-score
- it is closer to sentence- or chunk-level adaptive standardization with warm-started running stats

Useful implication for SSL:

- keep preprocessing standardization separate from learned session alignment
- if causal normalization is needed, use slow chunk/sentence adaptation with variance floors, not a fragile per-bin running std

## Session / Day Adaptation

One of the most consistent design choices across the Stanford speech-decoding family is explicit session adaptation.

In the `brain2text25` code:

- there is one day-specific input network per session
- the day-specific layer is part of the model, not just an offline preprocessing step

The model summary in the repo describes this as:

- `512 -> 512` day-specific linear transform
- `Softsign` nonlinearity
- one such layer per day / session

Why this matters:

- they do not assume feature standardization alone is enough
- they treat session drift and electrode-space mismatch as something worth learning explicitly
- the recurrent backbone is shared, but the input boundary is allowed to move with the day

For SSL on larger Utah-array data, this is one of the most transferable ideas.

## Sequence Front End

The local baseline uses temporal patching before the GRU:

- `patch_size = 14`
- `patch_stride = 4`

With `20 ms` bins, that means:

- each patch covers `280 ms`
- the effective step between patches is `80 ms`

This is a notable choice:

- they do not feed raw 20 ms bins directly into the recurrent core
- they reduce sequence length and add local temporal context before recurrence

This is useful to remember for SSL because:

- temporal patching may matter as much as backbone choice
- fairly aggressive patching is acceptable in a speech-motor setting

## Core Decoder Architecture

The local `brain2text25` baseline is:

- day-specific input layer(s)
- `5`-layer `GRU`
- hidden width `768`
- recurrent dropout `0.4`
- output linear layer to phoneme classes

This is a deliberately conservative architecture.

Important point:

- the performance story is not "use a huge foundation model"
- the performance story is "use a causally appropriate acoustic model, adapt it by day, then rely on strong LM decoding"

## Output Space And Loss

- output classes are phoneme logits
- the baseline vocabulary is `41` classes
- this includes:
  - `CTC blank`
  - phoneme labels
  - a silence / word-boundary token
- training uses `CTC`

Why this matters:

- the neural model is not asked to emit words directly
- it solves a cleaner acoustic-like subproblem first
- text generation quality is then heavily shaped by the external language model

For SSL planning, this argues for keeping representation learning and text generation somewhat decoupled.

## Smoothing

The local evaluation helper applies Gaussian smoothing to the neural features before the model forward pass.

In the local code:

- `smooth_kernel_std = 2`
- `smooth_kernel_size = 100`

This is another sign that the stack is intentionally simple:

- normalize
- smooth
- adapt by day
- recurrently decode phonemes

## Language Model Design

The text side is not built into the GRU.

Instead the full system uses:

- phoneme logits from the neural decoder
- an external n-gram language model
- optional later rescoring with `OPT`

The repo uses Redis to communicate with the external language model process during evaluation.

This separation matters conceptually:

- the neural model is optimized for phonetic / articulatory decoding
- the language model handles linguistic prior and sentence plausibility

For SSL notes, this suggests not overloading the encoder/backbone with responsibilities that may belong in a separate decoding stage.

## Data Mixing Across Days

The training setup mixes many sessions together while keeping:

- a shared recurrent model
- shared output head
- day-specific input layers

Batches may contain multiple days, and the model learns:

- what should be shared across days
- what should be isolated in day-specific parameters

That is a good reference design for multi-session SSL:

- do not collapse all heterogeneity into one shared input space if you can cheaply model boundary adaptation

## Calibration / Online Adaptation Philosophy

The Stanford system is not "train once and hope".

It explicitly uses:

- normalization that adapts over time
- day/session-specific input adaptation
- in some settings, online retraining

The overall philosophy is:

- keep the shared model reasonably stable
- adapt the interface to the current day
- fight nonstationarity continuously rather than pretending it is small

## Main Design Decisions Worth Carrying Forward

If I only wanted to remember a few choices from this stack for later Utah-array SSL work, they would be:

- use simple dense per-bin features first: `TX + SBP`
- treat nonstationarity handling as a separate problem from representation learning
- prefer slow adaptive normalization over fragile per-bin causal standardization
- include explicit session/day input adaptation
- consider temporal patching before the sequence backbone
- keep the neural backbone focused on phonetic / neural sequence modeling rather than full text generation
- do not underestimate the importance of the language-model stage

## Sources Used

- local repo notes and code:
  - `/Users/home/thesis/code/brain2text25/README.md`
  - `/Users/home/thesis/code/brain2text25/CLAUDE.md`
  - `/Users/home/thesis/code/brain2text25/model_training/rnn_args.yaml`
  - `/Users/home/thesis/code/brain2text25/model_training/dataset.py`
  - `/Users/home/thesis/code/brain2text25/model_training/evaluate_model_helpers.py`
  - `/Users/home/thesis/code/brain2text25/2024/competitionData/competitionData_readme.txt`
- public paper / patent sources consulted for the rolling-z-score description:
  - Nature 2023 speech neuroprosthesis paper
  - Stanford patent text describing rolling z-scoring warm-started from previous windows

## Limits Of These Notes

- the exact rolling-z-score blend coefficients are not present in the local Python files
- the public patent text makes the warm-start logic clear, but the OCR does not preserve every formula cleanly
- these notes should therefore be read as an accurate summary of the design direction, not as an exact implementation spec
