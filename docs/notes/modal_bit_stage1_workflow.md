# Modal BIT Stage-1 Workflow

This note records the current working setup for running the BIT-style stage-1
`S5` masked-reconstruction experiment on Modal.

It exists so future runs do not have to re-derive:

- which local data are canonical
- which Modal volumes exist
- how the volume filesystem should look
- which scripts are used
- which commands worked in practice

## Scope

This workflow is for the current generic `ssm_ssl` stage-1 run, not the full
three-stage BIT pipeline.

The active training script is:

- [scripts/modal/run_bit_s5_stage1.py](/Users/home/thesis/utah-ssl/scripts/modal/run_bit_s5_stage1.py)

Supporting Modal utilities are:

- [scripts/modal/create_utah_ssl_volumes.py](/Users/home/thesis/utah-ssl/scripts/modal/create_utah_ssl_volumes.py)
- [scripts/modal/extract_volume_archive.py](/Users/home/thesis/utah-ssl/scripts/modal/extract_volume_archive.py)
- [scripts/modal/recompute_bit_stage1_stats.py](/Users/home/thesis/utah-ssl/scripts/modal/recompute_bit_stage1_stats.py)
- [scripts/modal/run_bit_s5_stage2_ctc.py](/Users/home/thesis/utah-ssl/scripts/modal/run_bit_s5_stage2_ctc.py)

## Current Experiment Contract

This Modal path currently runs the BIT-style stage-1 configuration as:

- backbone: `S5`
- signal: explicit 256-channel TX with zero-padding for narrower datasets
- datasets and source splits: the exact named `BIT_STAGE1_DATASET_SPLITS` plan
  (`brain2text24/competition_test` is absent)
- boundary key mode: `session`
- cache root: smoothed cache
- stats: precomputed session-level stats
- patching: `patch_size=5`, `patch_stride=5`
- direction: `bidirectional`
- default steps: `60000`
- downstream CTC: off by default

Even though the original user request mentioned an `RTX 4090`, Modal's current
documented GPU offerings did not expose that exact device at implementation
time, so the script requests:

- `L40S` first
- `RTX-PRO-6000` fallback

## Canonical Local Sources

Use these local paths as the source of truth for the current Modal stage-1 run:

- full smoothed cache root:
  - `/Users/home/thesis/data/cache_v1_smoothed_sigma2p0`
- matching stats root:
  - `/Users/home/thesis/data/stats`

The relevant stats artifact is:

- `/Users/home/thesis/data/stats/session_feature_stats/smoothed_sigma2p0/tx_only/session/ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.pt`
- `/Users/home/thesis/data/stats/session_feature_stats/smoothed_sigma2p0/tx_only/session/ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.json`

## Modal Persistent Volumes

The workflow uses two named persistent Modal volumes:

- `utah-ssl-cache`
- `utah-ssl-outputs`

These are persistent named `modal.Volume` objects, not ephemeral scratch disks.

The helper script:

- [scripts/modal/create_utah_ssl_volumes.py](/Users/home/thesis/utah-ssl/scripts/modal/create_utah_ssl_volumes.py)

creates or verifies them and writes a tiny sentinel file.

Inside Modal containers, the mounts are:

- cache volume mounted at `/vol/cache`
- output volume mounted at `/vol/outputs`

## Expected Volume File Structure

The `utah-ssl-cache` volume should contain:

```text
/
  .volume_initialized.txt
  cache_v1_smoothed_sigma2p0/
    000950/
    brain2text24/
    motor_data/
    plug_n_play/
    unsupervised_cursor_recalibration_offline/
    unsupervised_cursor_recalibration_online/
    willett_handwriting/
  stats/
    session_feature_stats/
      smoothed_sigma2p0/
        tx_only/
          session/
            ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.pt
            ssl_pretrain_000950_brain2text24_motor_data_plug_n_play_unsupervised_cursor_recalibration_offline_unsupervised_cursor_recalibration_online_willett_handwriting_plan_f8843486db_v2.json
  uploads/
    ...
```

The `utah-ssl-outputs` volume will receive run outputs under:

```text
/ssl_experiments/modal_bit_s5_stage1/<run_name>/
```

## Recommended Upload Strategy

Uploading thousands of shard files directly to a Modal volume was unreliable in
practice on the user's connection. The stable approach was:

1. create tar archives locally
2. upload the tar files with `modal volume put`
3. extract them inside the volume with `extract_volume_archive.py`

### Why tar instead of zip

Use `tar`, not `zip`:

- fewer uploaded objects
- less per-file overhead
- simple extraction on Linux
- plain `.tar` avoids slow recompression/decompression

## Commands That Worked

### 1. Create or verify the persistent volumes

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/create_utah_ssl_volumes.py
```

If the volume already exists, `modal volume create` may report that; this is
fine.

### 2. Test extraction helper

The extractor script unpacks an archive already stored in the cache volume:

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/extract_volume_archive.py --archive /uploads/some_archive.tar --dest /
```

### 3. Upload `brain2text24` separately

This was used successfully as a smaller first test:

```bash
cd /Users/home/thesis/data
tar -cf brain2text24_smoothed_test.tar cache_v1_smoothed_sigma2p0/brain2text24
modal volume put utah-ssl-cache /Users/home/thesis/data/brain2text24_smoothed_test.tar /uploads/brain2text24_smoothed_test.tar
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/extract_volume_archive.py --archive /uploads/brain2text24_smoothed_test.tar --dest /
```

That leaves:

- `/cache_v1_smoothed_sigma2p0/brain2text24`

in the cache volume.

### 4. Upload the rest of the smoothed cache as a separate tar

Because `brain2text24` was already present, the rest of the datasets were
uploaded separately and extracted into the existing cache root:

```bash
cd /Users/home/thesis/data/cache_v1_smoothed_sigma2p0
tar -cf /Users/home/thesis/data/cache_v1_smoothed_rest.tar \
  000950 \
  motor_data \
  plug_n_play \
  unsupervised_cursor_recalibration_offline \
  unsupervised_cursor_recalibration_online \
  willett_handwriting

modal volume put utah-ssl-cache /Users/home/thesis/data/cache_v1_smoothed_rest.tar /uploads/cache_v1_smoothed_rest.tar

cd /Users/home/thesis/utah-ssl
modal run scripts/modal/extract_volume_archive.py --archive /uploads/cache_v1_smoothed_rest.tar --dest /cache_v1_smoothed_sigma2p0
```

This merged correctly with the already-uploaded `brain2text24` directory.

### 5. Upload the stage-1 stats

```bash
cd /Users/home/thesis/data
tar -cf stats_bit_stage1.tar stats/session_feature_stats/smoothed_sigma2p0/tx_only/session
modal volume put utah-ssl-cache /Users/home/thesis/data/stats_bit_stage1.tar /uploads/stats_bit_stage1.tar

cd /Users/home/thesis/utah-ssl
modal run scripts/modal/extract_volume_archive.py --archive /uploads/stats_bit_stage1.tar --dest /
```

### 6. Verify the cache volume contents

Useful checks:

```bash
modal volume ls utah-ssl-cache /
modal volume ls utah-ssl-cache /cache_v1_smoothed_sigma2p0
modal volume ls utah-ssl-cache /cache_v1_smoothed_sigma2p0/brain2text24
modal volume ls utah-ssl-cache /stats/session_feature_stats/smoothed_sigma2p0/tx_only/session
```

## Running The Actual Stage-1 Job

Once the cache and stats are in the volumes:

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/run_bit_s5_stage1.py
```

Or with an explicit run name:

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/run_bit_s5_stage1.py --run-name bit_s5_stage1_seed7_60k
```

The script will:

- mount `utah-ssl-cache` at `/vol/cache`
- mount `utah-ssl-outputs` at `/vol/outputs`
- read cache from:
  - `/vol/cache/cache_v1_smoothed_sigma2p0`
- expect stats under:
  - `/vol/cache/stats/session_feature_stats/smoothed_sigma2p0/tx_only/session`
- write outputs under:
  - `/vol/outputs/ssl_experiments/modal_bit_s5_stage1/<run_name>`

## Current Modal Script Defaults

At implementation time, the training script defaults are:

- `run_name = "bit_s5_stage1_l40s"`
- `ssl_steps = 60000`
- `ctc_steps = 12000`
- `run_downstream_ctc = False`
- `batch_size = 16`
- `hidden_size = 256`
- `state_size = 64`
- `num_layers = 4`
- `seed = 7`

## Known Failure Modes

### 1. Running from the wrong directory

If a command like:

```bash
modal run scripts/modal/create_utah_ssl_volumes.py
```

is launched from the wrong working directory, Modal may look for:

- `/Users/home/scripts/modal/...`

instead of the repo path.

Safe pattern:

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/...
```

### 2. Upload heartbeat or token expiry failures

Observed symptoms:

- local `Running app...` appears stuck
- progress freezes while elapsed time continues
- `Deadline exceeded`
- `Connection lost`
- `ExpiredToken`

Interpretation:

- the upload path was interrupted before Modal finalized it
- partially uploaded files should not be assumed usable

Response:

- verify with `modal volume ls`
- if the target path does not exist, treat that attempt as lost
- prefer tar uploads over huge per-file uploads

## Running A Clean Stage-2 Tx-Only CTC Fine-Tune

For the current generic `ssm_ssl` checkpoint family, use:

- [scripts/modal/run_bit_s5_stage2_ctc.py](/Users/home/thesis/utah-ssl/scripts/modal/run_bit_s5_stage2_ctc.py)

This stage-2 path:

- loads the `checkpoint_best.pt` from a prior stage-1 Modal run
- uses `brain2text24`
- keeps downstream input at `tx_only`
- disables extra online smoothing and noise augmentation by default when reading
  from the already-smoothed Modal cache, to avoid accidental double-smoothing

Example:

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/run_bit_s5_stage2_ctc.py --stage1-run-name bit_s5_stage1_seed7_60k
```

Optional random-init control in the same launch:

```bash
cd /Users/home/thesis/utah-ssl
modal run scripts/modal/run_bit_s5_stage2_ctc.py \
  --stage1-run-name bit_s5_stage1_seed7_60k \
  --also-random-init
```

### 3. Very slow direct per-file uploads

Uploading a large cache root as many individual files may degrade or wedge on
some connections. The tar-and-extract workflow is preferred.

## Recommended Next-Step Checklist

Before launching a future Modal stage-1 run:

1. verify `utah-ssl-cache` volume contains the smoothed cache root
2. verify the stage-1 stats artifact exists in the volume
3. verify `utah-ssl-outputs` exists
4. run `modal run scripts/modal/run_bit_s5_stage1.py`
5. inspect early logs to confirm:
   - cache path resolved correctly
   - stats path resolved correctly
   - training begins instead of failing at import or path discovery
