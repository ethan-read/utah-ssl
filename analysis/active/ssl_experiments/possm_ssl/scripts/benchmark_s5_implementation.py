#!/usr/bin/env python
"""Compare recurrent and FFT S5 training-step timings on synthetic tensors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch


def _ensure_repo_import_paths() -> None:
    script_dir = Path(__file__).resolve()
    repo_root = script_dir.parents[5]
    benchmark_dir = repo_root / "analysis" / "active" / "transfer_benchmark" / "ssl_autoresearch"
    for path in (repo_root, benchmark_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_import_paths()

from s5 import S5SequenceBackbone  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _step(
    model: S5SequenceBackbone,
    x: torch.Tensor,
    lengths: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    output = model(x, lengths)
    loss = (output * target).mean()
    loss.backward()
    if x.grad is None:
        raise RuntimeError("Input gradient was not populated.")
    return float(loss.detach().cpu()), output.detach(), x.grad.detach().clone()


def _time_steps(
    *,
    model: S5SequenceBackbone,
    x_base: torch.Tensor,
    lengths: torch.Tensor,
    target: torch.Tensor,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(int(warmup)):
        x = x_base.detach().clone().requires_grad_(True)
        _step(model, x, lengths, target)
    _sync(device)

    step_times: list[float] = []
    for _ in range(int(repeats)):
        x = x_base.detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        _step(model, x, lengths, target)
        _sync(device)
        step_times.append(time.perf_counter() - t0)
    return {
        "median_step_s": float(median(step_times)),
        "min_step_s": float(min(step_times)),
        "max_step_s": float(max(step_times)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-multiplier", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(str(args.device))
    recurrent = S5SequenceBackbone(
        d_model=int(args.d_model),
        d_state=int(args.d_state),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        ffn_multiplier=float(args.ffn_multiplier),
        implementation="recurrent",
    ).to(device)
    fft = S5SequenceBackbone(
        d_model=int(args.d_model),
        d_state=int(args.d_state),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        ffn_multiplier=float(args.ffn_multiplier),
        implementation="fft",
    ).to(device)
    fft.load_state_dict(recurrent.state_dict())
    recurrent.eval()
    fft.eval()

    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    if batch_size > 1:
        lengths[-1] = max(1, seq_len // 2)
    x_base = torch.randn(batch_size, seq_len, int(args.d_model), device=device)
    target = torch.randn_like(x_base)

    recurrent_x = x_base.detach().clone().requires_grad_(True)
    fft_x = x_base.detach().clone().requires_grad_(True)
    _, recurrent_output, recurrent_grad = _step(recurrent, recurrent_x, lengths, target)
    _, fft_output, fft_grad = _step(fft, fft_x, lengths, target)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    recurrent_timing = _time_steps(
        model=recurrent,
        x_base=x_base,
        lengths=lengths,
        target=target,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        device=device,
    )
    recurrent_peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fft_timing = _time_steps(
        model=fft,
        x_base=x_base,
        lengths=lengths,
        target=target,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        device=device,
    )
    fft_peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None

    payload = {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": int(args.d_model),
        "d_state": int(args.d_state),
        "num_layers": int(args.num_layers),
        "ffn_multiplier": float(args.ffn_multiplier),
        "dropout": float(args.dropout),
        "max_abs_output_diff": float((fft_output - recurrent_output).abs().max().cpu()),
        "max_abs_input_grad_diff": float((fft_grad - recurrent_grad).abs().max().cpu()),
        "recurrent": {**recurrent_timing, "peak_memory_mb": recurrent_peak_mb},
        "fft": {**fft_timing, "peak_memory_mb": fft_peak_mb},
    }
    rec_median = float(recurrent_timing["median_step_s"])
    fft_median = float(fft_timing["median_step_s"])
    payload["fft_speedup_vs_recurrent"] = rec_median / fft_median if fft_median > 0.0 else None
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
