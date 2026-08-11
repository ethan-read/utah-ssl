# RunPod Willett S5 TX+SBP Runbook

This is the first remote-GPU target for the supervised Willett-style S5
decoder. The research question is simple: does the stronger local S5 recipe
remain strong when run with the Willett-like area-6v `tx_sbp` feature set?

## Why This Run

Current notes say the supervised S5 `tx_only` run is the strongest baseline in
the corpus, with best visible PER around `0.336` at `12000` steps. A larger
area-6v `tx_sbp` run is the highest-value next rented-GPU job because it tests a
plain SSM decoder under the feature policy closest to the Willett reference
input.

Use this before broad SSL sweeps.

## Pod Choice

Start with one GPU, not a multi-GPU cluster.

Recommended order:

1. `A40` or `RTX A6000` if available cheaply. They have 48 GB VRAM and are good
   budget training cards for this model.
2. `RTX 4090` if the 48 GB cards are unavailable or slow.
3. `A100` only if the run is clearly too slow or memory-bound.

Use a PyTorch template. Keep the Pod alive with a start command like
`sleep infinity` while you are setting it up over SSH/Jupyter.

Suggested storage:

- container disk: `20-50 GB`
- pod volume disk: `50-100 GB`

The repo-local Brain2Text24 raw cache is about `2.3 GB`; checkpoints/logs for a
few runs should fit comfortably in this range.

If you are relying on the script to auto-stop the Pod, use normal Pod storage /
pod volume storage rather than a network volume. RunPod currently documents
that Pods with network volumes cannot be stopped, only terminated, so the
auto-stop API call may fail on network-volume Pods.

## Directory Layout On The Pod

Use `/workspace` for anything you want to survive a stopped Pod:

```text
/workspace/utah-ssl
/workspace/utah_ssl/data/cache_v1/brain2text24
/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp
```

The important cache check is:

```bash
ls /workspace/utah_ssl/data/cache_v1/brain2text24/manifest.jsonl
ls /workspace/utah_ssl/data/cache_v1/brain2text24/metadata.json
```

If using Google Drive or another transfer route, copy the Drive
`utah_ssl/data/cache_v1/brain2text24` folder into that cache root.

The launcher checks that this cache is the area-6v `tx+sbp` layout
(`128` TX + `128` SBP features). It also recomputes the canonical raw
`tx_sbp` global split stats on the Pod by default:

```text
/workspace/utah_ssl/data/stats/split_feature_stats/raw/brain2text24/competition_train/tx_sbp/global_v1.pt
```

That recompute is cheap compared with the run and avoids accidentally using
stale local or Drive stats with a different cache signature. Disable it only if
you deliberately copied matching stats too:

```bash
RECOMPUTE_SPLIT_STATS=0 experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh
```

## Auto-Stop Behavior

The launch script now tries to stop the Pod automatically after the training
command exits. This is meant for unattended runs when you do not want to babysit
the Pod after the job finishes or crashes.

It uses:

- `RUNPOD_POD_ID`
- `RUNPOD_API_KEY`

and sends:

```bash
POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop
```

`RUNPOD_POD_ID` should be available inside the Pod. You need to provide
`RUNPOD_API_KEY` yourself, either as a Pod environment variable/secret in the
RunPod UI or by exporting it in the shell before launch:

```bash
export RUNPOD_API_KEY="..."
```

Do this before starting the unattended `nohup` command. Without the API key, the
training job can still finish, but the Pod will not stop itself.

To keep the Pod alive on purpose, disable auto-stop for a run:

```bash
AUTO_STOP_POD_ON_EXIT=0 experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh
```

## Copy Data Onto The Pod

For this first run, the simplest path is to copy the prepared local cache from
your laptop to the Pod over SSH. The cache is small enough that this is
reasonable, and once the copy finishes your laptop can go offline.

In the RunPod UI, open the Pod connection panel and find:

- SSH host
- SSH port
- SSH username, usually `root`

Then run this from your laptop, replacing `HOST` and `PORT`:

```bash
ssh -p PORT root@HOST "mkdir -p /workspace/utah_ssl/data/cache_v1"

rsync -avP -e "ssh -p PORT" \
  /Users/home/thesis/utah-ssl/data/cache_v1/brain2text24 \
  root@HOST:/workspace/utah_ssl/data/cache_v1/
```

If your local canonical cache lives somewhere else, change the source path. The
destination should end up as:

```text
/workspace/utah_ssl/data/cache_v1/brain2text24/manifest.jsonl
/workspace/utah_ssl/data/cache_v1/brain2text24/metadata.json
/workspace/utah_ssl/data/cache_v1/brain2text24/shards/
```

If `rsync` is not installed inside the Pod, install it in the Pod terminal:

```bash
apt-get update
apt-get install -y rsync
```

Fallback without `rsync`:

```bash
tar -C /Users/home/thesis/utah-ssl/data/cache_v1 -czf - brain2text24 | \
  ssh -p PORT root@HOST \
  "mkdir -p /workspace/utah_ssl/data/cache_v1 && tar -C /workspace/utah_ssl/data/cache_v1 -xzf -"
```

After copying, verify from the Pod:

```bash
du -sh /workspace/utah_ssl/data/cache_v1/brain2text24
find /workspace/utah_ssl/data/cache_v1/brain2text24/shards -maxdepth 2 -type f | wc -l
ls /workspace/utah_ssl/data/cache_v1/brain2text24/manifest.jsonl
```

The local repo copy was about `2.3 GB` with roughly `221` shard files when this
runbook was written. Small differences are fine if the manifest and metadata
exist and the training script starts cleanly.

### Google Drive Alternative

If you prefer to avoid copying from your laptop, use `rclone` on the Pod to pull
from Google Drive into the same destination:

```bash
apt-get update
apt-get install -y rclone
rclone config
rclone copy "gdrive:utah_ssl/data/cache_v1/brain2text24" \
  /workspace/utah_ssl/data/cache_v1/brain2text24 \
  --progress
```

This is nice once configured, but the browser/OAuth step can be annoying on a
remote machine. For a first RunPod run, SSH/`rsync` from the laptop is usually
less mysterious.

## One-Time Setup

From a terminal on the Pod:

```bash
cd /workspace
git clone <YOUR_REPO_URL> utah-ssl
cd /workspace/utah-ssl
python -m pip install -r experiments/supervised_baselines/launchers/runpod/requirements.txt
python -m unittest experiments.supervised_baselines.tests.test_willett_reconstruction -q
```

Do not `pip install torch` unless the selected image lacks PyTorch. The RunPod
PyTorch image should already provide a CUDA-compatible `torch`.

Before using `git clone` on the Pod, make sure the local RunPod scripts and CV
changes are committed and pushed to the repo/branch you clone. Otherwise the Pod
will not see this runbook's launch scripts.

## Launch The Main Run

From `/workspace/utah-ssl`:

```bash
chmod +x experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh

CACHE_ROOT=/workspace/utah_ssl/data/cache_v1 \
OUTPUT_ROOT=/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp \
RUN_NAME=willett_s5_tx_sbp_seed7_60k \
MAX_STEPS=60000 \
SEED=7 \
experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh
```

For an unattended run that keeps going after SSH disconnects:

```bash
cd /workspace/utah-ssl
mkdir -p /workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp
nohup env \
  CACHE_ROOT=/workspace/utah_ssl/data/cache_v1 \
  OUTPUT_ROOT=/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp \
  RUN_NAME=willett_s5_tx_sbp_seed7_60k \
  MAX_STEPS=60000 \
  SEED=7 \
  experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh \
  > /workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp/seed7_60k.nohup.log 2>&1 &
```

## Optional Cross-Validation

Use cross-validation to estimate run-to-run and split sensitivity before or
alongside the official `competition_train -> competition_test` benchmark. The
CV split is made only inside `competition_train`; `competition_test` remains the
final benchmark split.

The fold policy is deterministic and within-session: each fold holds out part
of every released training session, so session adapters are represented in both
train and validation.

Run one fold:

```bash
CACHE_ROOT=/workspace/utah_ssl/data/cache_v1 \
OUTPUT_ROOT=/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp_cv \
SPLIT_POLICY=competition_train_kfold \
CV_NUM_FOLDS=5 \
CV_FOLD_INDEX=0 \
RUN_NAME=willett_s5_tx_sbp_cv5_seed7_60k_fold0 \
MAX_STEPS=60000 \
SEED=7 \
experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh
```

Run all folds sequentially, then auto-stop the Pod:

```bash
nohup env \
  CACHE_ROOT=/workspace/utah_ssl/data/cache_v1 \
  OUTPUT_ROOT=/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp_cv \
  CV_NUM_FOLDS=5 \
  RUN_NAME_PREFIX=willett_s5_tx_sbp_cv5_seed7_60k \
  MAX_STEPS=60000 \
  SEED=7 \
  experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp_cross_validation.sh \
  > /workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp_cv/cv5_seed7_60k.nohup.log 2>&1 &
```

For CV, the trainer computes `global` z-scoring stats from each fold's training
rows in memory. It does not use the official `competition_train` split-stats
artifact, because that would leak each fold's held-out rows into normalization.

## Monitor

```bash
tail -f /workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp/seed7_60k.nohup.log
tail -f /workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp/willett_s5_tx_sbp_seed7_60k/progress.jsonl
```

Checkpoint and summary files land under:

```text
/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp/willett_s5_tx_sbp_seed7_60k/
```

Key files:

- `progress.jsonl`
- `summary.json`
- `checkpoint_best.pt`
- `checkpoint_final.pt`
- `checkpoints/step_*.pt`

## Resume

The launch script passes `--resume-latest`, so rerunning the same command with
the same `OUTPUT_ROOT` and `RUN_NAME` resumes from the latest checkpoint.

To extend a completed 60k run to 80k:

```bash
MAX_STEPS=80000 \
RUN_NAME=willett_s5_tx_sbp_seed7_60k \
experiments/supervised_baselines/launchers/runpod/train_s5_tx_sbp.sh
```

The runner will keep the same run directory and continue.

## Stop/Cost Hygiene

Because the launcher auto-stops the Pod after the training command exits, the
main thing to protect is persistence of `/workspace`. Use a volume or network
volume if you want the outputs to survive after the Pod stops. Terminate the Pod
only after you are sure the outputs are saved somewhere persistent.
