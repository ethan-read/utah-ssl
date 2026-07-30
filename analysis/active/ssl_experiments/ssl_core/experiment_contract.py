"""Explicit signal, dataset, and experiment contracts for SSL workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .feature_contract import FeatureContract, resolve_feature_contract


@dataclass(frozen=True)
class SignalSpec:
    """The exact neural signal view consumed by an analysis or model."""

    mode: str
    tx_dim: int = 0
    sbp_dim: int = 0
    column_start: int = 0
    missing_channel_policy: str = "error"

    def __post_init__(self) -> None:
        contract = resolve_feature_contract(self.mode)
        if int(self.column_start) < 0:
            raise ValueError("column_start must be non-negative")
        if self.missing_channel_policy not in {"error", "zero_pad"}:
            raise ValueError("missing_channel_policy must be one of {'error', 'zero_pad'}")
        if contract.uses_tx and int(self.tx_dim) <= 0:
            raise ValueError("tx_dim must be positive when the signal uses TX")
        if not contract.uses_tx and int(self.tx_dim) != 0:
            raise ValueError("tx_dim must be zero when the signal does not use TX")
        if contract.uses_sbp and int(self.sbp_dim) <= 0:
            raise ValueError("sbp_dim must be positive when the signal uses SBP")
        if not contract.uses_sbp and int(self.sbp_dim) != 0:
            raise ValueError("sbp_dim must be zero when the signal does not use SBP")

    @property
    def contract(self) -> FeatureContract:
        return resolve_feature_contract(self.mode)

    @property
    def modalities(self) -> tuple[str, ...]:
        return self.contract.modalities

    @property
    def full_dim(self) -> int:
        return self.contract.full_dim(tx_dim=int(self.tx_dim), sbp_dim=int(self.sbp_dim))

    def required_dim(self, modality: str) -> int:
        name = str(modality)
        if name == "tx":
            return int(self.tx_dim)
        if name == "sbp":
            return int(self.sbp_dim)
        raise ValueError(f"Unsupported signal modality: {name!r}")

    def selected_columns(self, modality: str) -> tuple[int, int]:
        if str(modality) not in self.modalities:
            raise ValueError(f"Signal mode {self.mode!r} does not use {modality!r}")
        start = int(self.column_start)
        return start, start + self.required_dim(str(modality))

    def selected_columns_for_width(
        self,
        modality: str,
        available_width: int,
    ) -> tuple[int, int]:
        """Resolve a physical slice while enforcing the missing-channel policy."""
        start, requested_stop = self.selected_columns(modality)
        width = int(available_width)
        if width < 0:
            raise ValueError("available_width must be non-negative")
        if self.missing_channel_policy == "error" and width < requested_stop:
            raise ValueError(
                f"Signal contract requests {modality} columns [{start}, {requested_stop}), "
                f"but the shard contains only {width} columns."
            )
        return start, min(requested_stop, width)

    def row_is_compatible(
        self,
        *,
        has_tx: bool,
        has_sbp: bool,
        n_tx_features: int,
        n_sbp_features: int,
    ) -> bool:
        if self.missing_channel_policy == "zero_pad":
            required_tx = int(self.column_start) + 1 if self.contract.uses_tx else 0
            required_sbp = int(self.column_start) + 1 if self.contract.uses_sbp else 0
        else:
            required_tx = (
                int(self.column_start) + int(self.tx_dim)
                if self.contract.uses_tx
                else 0
            )
            required_sbp = (
                int(self.column_start) + int(self.sbp_dim)
                if self.contract.uses_sbp
                else 0
            )
        return self.contract.row_is_compatible(
            has_tx=has_tx,
            has_sbp=has_sbp,
            n_tx_features=n_tx_features,
            n_sbp_features=n_sbp_features,
            tx_dim=required_tx,
            sbp_dim=required_sbp,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(
        cls,
        value: "SignalSpec | Mapping[str, Any]",
    ) -> "SignalSpec":
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError("signal_spec must be a SignalSpec or mapping")
        return cls(**dict(value))

    @classmethod
    def from_mode(
        cls,
        mode: str,
        *,
        tx_dim: int,
        sbp_dim: int,
        column_start: int = 0,
        missing_channel_policy: str = "error",
    ) -> "SignalSpec":
        contract = resolve_feature_contract(mode)
        return cls(
            mode=contract.mode,
            tx_dim=int(tx_dim) if contract.uses_tx else 0,
            sbp_dim=int(sbp_dim) if contract.uses_sbp else 0,
            column_start=int(column_start),
            missing_channel_policy=str(missing_channel_policy),
        )

    @classmethod
    def tx_only(
        cls,
        *,
        tx_dim: int,
        column_start: int = 0,
        missing_channel_policy: str = "error",
    ) -> "SignalSpec":
        return cls(
            mode="tx_only",
            tx_dim=int(tx_dim),
            sbp_dim=0,
            column_start=int(column_start),
            missing_channel_policy=str(missing_channel_policy),
        )

    @classmethod
    def sbp_only(
        cls,
        *,
        sbp_dim: int,
        column_start: int = 0,
        missing_channel_policy: str = "error",
    ) -> "SignalSpec":
        return cls(
            mode="sbp_only",
            tx_dim=0,
            sbp_dim=int(sbp_dim),
            column_start=int(column_start),
            missing_channel_policy=str(missing_channel_policy),
        )

    @classmethod
    def tx_sbp(
        cls,
        *,
        tx_dim: int,
        sbp_dim: int,
        column_start: int = 0,
        missing_channel_policy: str = "error",
    ) -> "SignalSpec":
        return cls(
            mode="tx_sbp",
            tx_dim=int(tx_dim),
            sbp_dim=int(sbp_dim),
            column_start=int(column_start),
            missing_channel_policy=str(missing_channel_policy),
        )


@dataclass(frozen=True)
class DatasetSelection:
    name: str
    source_splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("dataset name must be non-empty")
        splits = tuple(
            sorted(
                {
                    str(split).strip().lower()
                    for split in self.source_splits
                    if str(split).strip()
                }
            )
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_splits", splits)


@dataclass(frozen=True)
class DatasetPlan:
    """An explicit list of datasets and allowed source splits."""

    datasets: tuple[DatasetSelection, ...]

    def __post_init__(self) -> None:
        normalized: list[DatasetSelection] = []
        seen: set[str] = set()
        for value in self.datasets:
            item = value if isinstance(value, DatasetSelection) else DatasetSelection(**dict(value))
            if item.name in seen:
                raise ValueError(f"dataset plan contains duplicate dataset {item.name!r}")
            seen.add(item.name)
            normalized.append(item)
        if not normalized:
            raise ValueError("dataset plan must contain at least one dataset")
        object.__setattr__(self, "datasets", tuple(sorted(normalized, key=lambda item: item.name)))

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.datasets)

    @property
    def source_splits_by_dataset(self) -> dict[str, tuple[str, ...]]:
        return {
            item.name: item.source_splits
            for item in self.datasets
            if item.source_splits
        }

    def to_dict(self) -> dict[str, list[str]]:
        return {
            item.name: list(item.source_splits)
            for item in self.datasets
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Sequence[str]],
    ) -> "DatasetPlan":
        return cls(
            tuple(
                DatasetSelection(name=str(dataset), source_splits=tuple(splits))
                for dataset, splits in value.items()
            )
        )

    @classmethod
    def from_value(
        cls,
        value: "DatasetPlan | Mapping[str, Sequence[str]]",
    ) -> "DatasetPlan":
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError("dataset_plan must be a DatasetPlan or mapping")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class ExperimentRecipe:
    """A named, serializable experiment boundary."""

    name: str
    dataset_plan: DatasetPlan
    signal_spec: SignalSpec
    description: str = ""
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("recipe name must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dataset_plan", DatasetPlan.from_value(self.dataset_plan))
        object.__setattr__(self, "signal_spec", SignalSpec.from_value(self.signal_spec))
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": str(self.description),
            "dataset_plan": self.dataset_plan.to_dict(),
            "signal_spec": self.signal_spec.to_dict(),
            "settings": dict(self.settings),
        }


__all__ = [
    "DatasetPlan",
    "DatasetSelection",
    "ExperimentRecipe",
    "SignalSpec",
]
