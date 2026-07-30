"""Shared feature-layout contracts for cache-backed SSL and POSSM workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureContract:
    """Canonical meaning of one cache feature mode."""

    mode: str
    modalities: tuple[str, ...]

    @property
    def uses_tx(self) -> bool:
        return "tx" in self.modalities

    @property
    def uses_sbp(self) -> bool:
        return "sbp" in self.modalities

    def full_dim(self, *, tx_dim: int, sbp_dim: int) -> int:
        return (
            (int(tx_dim) if self.uses_tx else 0)
            + (int(sbp_dim) if self.uses_sbp else 0)
        )

    def feature_start(self, modality: str, *, tx_dim: int) -> int:
        name = str(modality)
        if name not in self.modalities:
            raise ValueError(
                f"Feature mode {self.mode!r} does not include modality {name!r}"
            )
        if name == "sbp" and self.uses_tx:
            return int(tx_dim)
        return 0

    def row_is_compatible(
        self,
        *,
        has_tx: bool,
        has_sbp: bool,
        n_tx_features: int,
        n_sbp_features: int,
        tx_dim: int,
        sbp_dim: int,
    ) -> bool:
        if self.uses_tx and (
            not bool(has_tx) or int(n_tx_features) < int(tx_dim)
        ):
            return False
        if self.uses_sbp and (
            not bool(has_sbp) or int(n_sbp_features) < int(sbp_dim)
        ):
            return False
        return True


_FEATURE_CONTRACTS = {
    "tx_only": FeatureContract(
        mode="tx_only",
        modalities=("tx",),
    ),
    "sbp_only": FeatureContract(
        mode="sbp_only",
        modalities=("sbp",),
    ),
    "tx_sbp": FeatureContract(
        mode="tx_sbp",
        modalities=("tx", "sbp"),
    ),
}

SUPPORTED_FEATURE_MODES = tuple(_FEATURE_CONTRACTS)


def resolve_feature_contract(feature_mode: str) -> FeatureContract:
    mode = str(feature_mode)
    try:
        return _FEATURE_CONTRACTS[mode]
    except KeyError as exc:
        raise ValueError(
            f"feature_mode must be one of {SUPPORTED_FEATURE_MODES}; got {mode!r}"
        ) from exc


def modalities_for_feature_mode(feature_mode: str) -> tuple[str, ...]:
    return resolve_feature_contract(feature_mode).modalities


__all__ = [
    "FeatureContract",
    "SUPPORTED_FEATURE_MODES",
    "modalities_for_feature_mode",
    "resolve_feature_contract",
]
