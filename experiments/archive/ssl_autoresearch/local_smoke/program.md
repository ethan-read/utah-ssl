# ssl_autoresearch_local

This directory is a local smoke-test benchmark for autoresearch-style SSL experiments.

The purpose is to validate the automated research loop on a laptop before building the full CUDA-dependent pipeline.

## Setup

Before beginning an autonomous loop:

1. work on a dedicated git branch for this run
2. read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `data.py`
   - `train.py`
   - `run_experiment.py`
   - `search_space.py`
3. verify that the local Brain2Text25 `TX` and `SBP` caches exist
4. initialize `results.tsv` if it does not already exist by running one experiment through `run_experiment.py`

Do not begin an indefinite loop on top of an unrelated dirty worktree.

## Goal

Minimize `val_ssl_loss`, the validation self-supervised loss on held-out sessions reported by `train.py` after a fixed `5` minute training budget.

This is a temporary local benchmark metric.

It is useful for:

- validating the automation loop
- comparing local smoke-test variants
- checking whether code changes improve the fixed local SSL task

It is not yet the final thesis metric.

## Editable Surface

You may modify:

- `train.py`

You may vary:

- architecture details implemented in `train.py`
- optimization hyperparameters exposed through `run_experiment.py`
- any configuration that remains inside the explicit search space defined in `search_space.py`

## Frozen Benchmark Contract

Do not modify during experiments:

- `prepare.py`
- `data.py`
- `run_experiment.py`
- the underlying cache files in `code/ssl/`

Keep the benchmark cheap and auditable.

Prefer simpler changes when gains are similar.

## Important Constraint

This is a local plumbing benchmark, not yet the final thesis model benchmark.

That means:

- the current metric is temporary
- the current model in `train.py` is only a small causal placeholder
- successful local smoke results should inform a later CUDA pipeline, not be treated as final architecture evidence

## Required Summary Output

`train.py` must print a summary block containing at least:

```text
val_ssl_loss:
training_seconds:
total_seconds:
device:
num_steps:
num_params:
patch_size:
patch_stride:
hidden_size:
num_layers:
standardize_scope:
post_proj_norm:
```

## Logging

Log every run to `results.tsv`.

Columns:

```text
timestamp	patch_size	patch_stride	hidden_size	num_layers	learning_rate	batch_size	standardize_scope	post_proj_norm	val_ssl_loss	status	note
```

Status should be:

- `keep` if `val_ssl_loss` is strictly lower than the best earlier successful run
- `discard` otherwise
- `crash` if the run fails to produce a summary block

Never commit `results.tsv`.

## Search Discipline

Stay inside the search space in `search_space.py`.

For the local smoke benchmark, allowed search should be concentrated on:

- `patch_size`
- `patch_stride`
- `hidden_size`
- `num_layers`
- `learning_rate`
- `batch_size`
- `standardize_scope`
- `post_proj_norm`

Architecture edits inside `train.py` are allowed, but they should remain consistent with the benchmark’s role as a small causal smoke test.

## Experiment Loop

LOOP FOREVER:

1. Check git state and identify the current kept commit on the dedicated run branch.
2. Choose one experimental idea within the benchmark scope.
3. Modify `train.py` directly if needed.
4. Commit the experimental change.
5. Run the experiment through the wrapper, redirecting output to a log file. For example:

```bash
python run_experiment.py --patch-size 1 --patch-stride 1 --hidden-size 64 --num-layers 1 --learning-rate 3e-4 --batch-size 16 --standardize-scope subject --post-proj-norm rms --note "baseline" > run.log 2>&1
```

6. Read out the result from the log file:

```bash
grep "^val_ssl_loss:\|^device:\|^num_steps:" run.log
```

7. If the summary block is missing, treat the run as a crash. Read the traceback:

```bash
tail -n 50 run.log
```

8. Record the result in `results.tsv`. The wrapper already appends one row and assigns `keep`, `discard`, or `crash`.
9. If the run improved `val_ssl_loss`, advance the branch by keeping the commit.
10. If the run did not improve, discard the experimental commit and return the branch to the previous kept commit.

The basic idea is:

- good changes stay on the branch
- bad changes are discarded
- the branch should represent the current best local benchmark recipe

## Timeout Policy

- the fixed training budget inside `train.py` is `5` minutes
- the outer wrapper allows additional startup and evaluation overhead
- a normal run should finish in under about `10` minutes total

If a run exceeds the wrapper timeout or otherwise hangs substantially past the intended budget, treat it as a failure.

## Crash Policy

If a run crashes:

- fix obvious implementation mistakes and retry if the idea still makes sense
- if the idea itself appears broken, log the crash and move on
- do not get stuck repeatedly debugging one bad idea at the expense of the loop

## Autonomy

Once the experiment loop has begun, do not pause to ask whether to continue.

Do not ask:

- whether this is a good stopping point
- whether one more run is desired
- whether the human still wants the loop to continue

The expectation is autonomous iteration until the human interrupts the process.

If progress stalls:

- re-read `README.md`
- re-read `train.py`
- re-read `search_space.py`
- prefer small controlled changes before radical rewrites
- only attempt more aggressive design changes after the simpler search region is exhausted
