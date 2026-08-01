"""Precision, optimizer, and timing helpers for POSSM training."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import torch


SUPPORTED_PRECISIONS = ("float32", "amp_fp16")


def validate_precision(value: str) -> str:
    precision = str(value)
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of {set(SUPPORTED_PRECISIONS)}")
    return precision


@dataclass
class PrecisionRuntime:
    requested: str
    resolved: str
    device: torch.device
    scaler: torch.amp.GradScaler

    @property
    def amp_enabled(self) -> bool:
        return self.resolved == "amp_fp16"

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return torch.float16 if self.amp_enabled else None

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_precision": self.requested,
            "resolved_precision": self.resolved,
            "amp_enabled": bool(self.amp_enabled),
            "autocast_dtype": (
                str(self.autocast_dtype).removeprefix("torch.")
                if self.autocast_dtype is not None
                else None
            ),
            "grad_scaler_enabled": bool(self.scaler.is_enabled()),
            "grad_scaler_scale": (
                float(self.scaler.get_scale()) if self.scaler.is_enabled() else None
            ),
        }


def resolve_precision_runtime(
    precision: str,
    *,
    device: torch.device,
) -> PrecisionRuntime:
    requested = validate_precision(precision)
    resolved_device = torch.device(device)
    if requested == "amp_fp16":
        if resolved_device.type != "cuda":
            raise RuntimeError(
                "precision='amp_fp16' requires a CUDA device; "
                f"received device={resolved_device}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "precision='amp_fp16' requires an available CUDA runtime and GPU, "
                "but torch.cuda.is_available() is False."
            )
        device_index = resolved_device.index
        if device_index is not None and device_index >= torch.cuda.device_count():
            raise RuntimeError(
                "precision='amp_fp16' received an unavailable CUDA device: "
                f"device={resolved_device}, device_count={torch.cuda.device_count()}."
            )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=requested == "amp_fp16",
    )
    return PrecisionRuntime(
        requested=requested,
        resolved=requested,
        device=resolved_device,
        scaler=scaler,
    )


def autocast_context(runtime: PrecisionRuntime):
    if not runtime.amp_enabled:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    )


def build_adamw(
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    *,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.optim.AdamW, bool]:
    resolved_params = list(params)
    kwargs: dict[str, Any] = {
        "lr": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    use_fused = torch.device(device).type == "cuda"
    if use_fused:
        kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(resolved_params, **kwargs)
    except (RuntimeError, TypeError):
        if not use_fused:
            raise
        kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(resolved_params, **kwargs)
        use_fused = False
    return optimizer, bool(use_fused)


class PhaseTimer:
    """Collect phase timings without synchronizing until ``finish`` is called."""

    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self._cuda_enabled = self.enabled and self.device.type == "cuda"
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._cpu_seconds: dict[str, float] = {}

    def start(self) -> torch.cuda.Event | float | None:
        if not self.enabled:
            return None
        if self._cuda_enabled:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, phase: str, token: torch.cuda.Event | float | None) -> None:
        if token is None or not self.enabled:
            return
        if self._cuda_enabled:
            if not isinstance(token, torch.cuda.Event):
                raise TypeError("CUDA PhaseTimer received a non-CUDA start token")
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._events.setdefault(str(phase), []).append((token, end))
            return
        self._cpu_seconds[str(phase)] = self._cpu_seconds.get(str(phase), 0.0) + (
            time.perf_counter() - float(token)
        )

    def finish(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        result = dict(self._cpu_seconds)
        if self._cuda_enabled:
            torch.cuda.synchronize(self.device)
            for phase, pairs in self._events.items():
                result[phase] = sum(
                    float(start.elapsed_time(end)) / 1000.0
                    for start, end in pairs
                )
        return result
