"""Run generic S5/Mamba SSL plus downstream CTC controls.

Example Colab usage:

python experiments/bit_style/scripts/run_generic_ssm_ssl.py \
  --cache-root /content/drive/MyDrive/utah_ssl/data/cache_v1 \
  --output-root /content/drive/MyDrive/utah_ssl/outputs/ssl_experiments/ssm_ssl \
  --backbone-type s5 --input-mode temporal_patch --ssl-steps 8000 --ctc-steps 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bit_style.config import GenericSSMSSLConfig
from experiments.bit_style.training import run_generic_ssm_ssl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--backbone-type", choices=("s5", "mamba"), default=None)
    parser.add_argument("--input-mode", choices=("raw_bin", "temporal_patch", "causal_conv_stem"), default=None)
    parser.add_argument("--ssl-steps", type=int, default=None)
    parser.add_argument("--ctc-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--state-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--mask-time-ratio", type=float, default=None)
    parser.add_argument("--mask-channel-ratio", type=float, default=None)
    parser.add_argument("--no-downstream-ctc", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> GenericSSMSSLConfig:
    payload = {}
    if args.config_json is not None:
        payload.update(json.loads(Path(args.config_json).read_text()))
    overrides = {
        "cache_root": args.cache_root,
        "output_root": args.output_root,
        "run_name": args.run_name,
        "backbone_type": args.backbone_type,
        "input_mode": args.input_mode,
        "ssl_steps": args.ssl_steps,
        "ctc_steps": args.ctc_steps,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "state_size": args.state_size,
        "num_layers": args.num_layers,
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "mask_time_ratio": args.mask_time_ratio,
        "mask_channel_ratio": args.mask_channel_ratio,
    }
    payload.update({key: value for key, value in overrides.items() if value is not None})
    if args.no_downstream_ctc:
        payload["run_downstream_ctc"] = False
    return GenericSSMSSLConfig.from_dict(payload)


def main() -> None:
    config = _config_from_args(_parse_args())
    summary = run_generic_ssm_ssl(config)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
