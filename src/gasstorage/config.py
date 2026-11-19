"""Typed configuration objects loaded from YAML.

Every section of ``configs/*.yaml`` maps to a frozen dataclass. Construction validates
the values, so downstream code can rely on them without re-checking. Unknown and missing
keys are rejected so that a typo in the YAML never silently falls back to a default.

Contract parameters are illustrative and do not describe a real storage site.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

TimeStep = Literal["daily", "weekly"]
CurveSource = Literal["synthetic", "csv"]
ModelName = Literal["one_factor", "two_factor"]
BasisName = Literal["polynomial", "laguerre"]

_TOL = 1e-12


def _check_keys(cls: type, raw: Mapping[str, Any], section: str) -> None:
    expected = {f.name for f in dataclasses.fields(cls)}
    given = set(raw)
    if unknown := given - expected:
        raise ValueError(f"[{section}] unknown key(s): {sorted(unknown)}")
    if missing := expected - given:
        raise ValueError(f"[{section}] missing key(s): {sorted(missing)}")


def _as_mapping(raw: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{section}] expected a mapping, got {type(raw).__name__}")
    return raw


def _as_date(value: Any, name: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name}: expected an ISO date YYYY-MM-DD, got {value!r}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: float, name: str) -> None:
    _require(math.isfinite(value), f"{name} must be finite, got {value}")


@dataclass(frozen=True, slots=True)
class RatchetTier:
    r"""Rate multiplier applied when the fill level $v/\bar V$ lies in ``[lower, upper)``.

    The effective maximum rate at fill level $\ell$ is
    $q^{\max}(\ell) = q^{\max} \cdot \sum_k \phi_k \, \mathbf 1\{\ell \in [a_k, b_k)\}$,
    where the tiers $[a_k, b_k)$ partition $[0, 1]$ (the last tier is closed at 1).
    """

    lower: float
    upper: float
    factor: float


def _parse_tiers(raw: Any, name: str) -> tuple[RatchetTier, ...]:
    _require(isinstance(raw, list) and len(raw) > 0, f"{name}: need at least one ratchet tier")
    tiers = []
    for i, item in enumerate(raw):
        item = _as_mapping(item, f"{name}[{i}]")
        _check_keys(RatchetTier, item, f"{name}[{i}]")
        tiers.append(RatchetTier(float(item["lower"]), float(item["upper"]), float(item["factor"])))
    tiers.sort(key=lambda t: t.lower)

    _require(
        abs(tiers[0].lower) < _TOL and abs(tiers[-1].upper - 1.0) < _TOL,
        f"{name}: tiers must cover the fill range [0, 1]",
    )
    for prev, nxt in itertools.pairwise(tiers):
        _require(abs(prev.upper - nxt.lower) < _TOL, f"{name}: tiers must be contiguous")
    for t in tiers:
        _require(t.lower < t.upper, f"{name}: empty tier [{t.lower}, {t.upper})")
        _require(0.0 < t.factor <= 1.0, f"{name}: factor must be in (0, 1], got {t.factor}")
    return tuple(tiers)


@dataclass(frozen=True, slots=True)
class ContractConfig:
    """Storage contract terms. Volumes in MWh, costs in EUR/MWh, rates in MWh/day."""

    start_date: dt.date
    end_date: dt.date
    working_volume_mwh: float
    initial_inventory_mwh: float
    terminal_inventory_mwh: float
    terminal_penalty_eur_mwh: float | None
    injection_days_to_fill: float
    withdrawal_days_to_empty: float
    injection_ratchets: tuple[RatchetTier, ...]
    withdrawal_ratchets: tuple[RatchetTier, ...]
    injection_cost_eur_mwh: float
    withdrawal_cost_eur_mwh: float
    fuel_loss: float

    def __post_init__(self) -> None:
        _require(self.end_date > self.start_date, "end_date must be after start_date")
        _require(self.working_volume_mwh > 0, "working_volume_mwh must be > 0")
        for name in ("initial_inventory_mwh", "terminal_inventory_mwh"):
            v = getattr(self, name)
            _require(0 <= v <= self.working_volume_mwh, f"{name} must be in [0, working volume]")
        if self.terminal_penalty_eur_mwh is not None:
            _require(self.terminal_penalty_eur_mwh >= 0, "terminal_penalty_eur_mwh must be >= 0")
        _require(self.injection_days_to_fill > 0, "injection_days_to_fill must be > 0")
        _require(self.withdrawal_days_to_empty > 0, "withdrawal_days_to_empty must be > 0")
        _require(self.injection_cost_eur_mwh >= 0, "injection_cost_eur_mwh must be >= 0")
        _require(self.withdrawal_cost_eur_mwh >= 0, "withdrawal_cost_eur_mwh must be >= 0")
        _require(0 <= self.fuel_loss < 1, "fuel_loss must be in [0, 1)")

    @property
    def max_injection_mwh_per_day(self) -> float:
        r"""Unratcheted maximum injection rate $\bar V / d_{\text{inj}}$."""
        return self.working_volume_mwh / self.injection_days_to_fill

    @property
    def max_withdrawal_mwh_per_day(self) -> float:
        r"""Unratcheted maximum withdrawal rate $\bar V / d_{\text{wdr}}$."""
        return self.working_volume_mwh / self.withdrawal_days_to_empty

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ContractConfig:
        raw = _as_mapping(raw, "contract")
        _check_keys(cls, raw, "contract")
        penalty = raw["terminal_penalty_eur_mwh"]
        return cls(
            start_date=_as_date(raw["start_date"], "start_date"),
            end_date=_as_date(raw["end_date"], "end_date"),
            working_volume_mwh=float(raw["working_volume_mwh"]),
            initial_inventory_mwh=float(raw["initial_inventory_mwh"]),
            terminal_inventory_mwh=float(raw["terminal_inventory_mwh"]),
            terminal_penalty_eur_mwh=None if penalty is None else float(penalty),
            injection_days_to_fill=float(raw["injection_days_to_fill"]),
            withdrawal_days_to_empty=float(raw["withdrawal_days_to_empty"]),
            injection_ratchets=_parse_tiers(raw["injection_ratchets"], "injection_ratchets"),
            withdrawal_ratchets=_parse_tiers(raw["withdrawal_ratchets"], "withdrawal_ratchets"),
            injection_cost_eur_mwh=float(raw["injection_cost_eur_mwh"]),
            withdrawal_cost_eur_mwh=float(raw["withdrawal_cost_eur_mwh"]),
            fuel_loss=float(raw["fuel_loss"]),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCurveConfig:
    """Parameters of the SYNTHETIC seasonal forward curve (not market data)."""

    base_level_eur_mwh: float
    winter_premium_eur_mwh: float
    contango_eur_mwh_per_year: float

    def __post_init__(self) -> None:
        _require(self.base_level_eur_mwh > 0, "base_level_eur_mwh must be > 0")
        _require(self.winter_premium_eur_mwh >= 0, "winter_premium_eur_mwh must be >= 0")
        _finite(self.contango_eur_mwh_per_year, "contango_eur_mwh_per_year")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SyntheticCurveConfig:
        raw = _as_mapping(raw, "market.forward_curve.synthetic")
        _check_keys(cls, raw, "market.forward_curve.synthetic")
        return cls(**{k: float(v) for k, v in raw.items()})


@dataclass(frozen=True, slots=True)
class ForwardCurveConfig:
    """Where the initial forward curve $F(0, T)$ comes from."""

    source: CurveSource
    csv_path: Path | None
    synthetic: SyntheticCurveConfig

    def __post_init__(self) -> None:
        _require(self.source in ("synthetic", "csv"), f"unknown curve source {self.source!r}")
        if self.source == "csv":
            _require(self.csv_path is not None, "csv_path is required when source is 'csv'")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ForwardCurveConfig:
        raw = _as_mapping(raw, "market.forward_curve")
        _check_keys(cls, raw, "market.forward_curve")
        path = raw["csv_path"]
        return cls(
            source=raw["source"],
            csv_path=None if path is None else Path(path),
            synthetic=SyntheticCurveConfig.from_dict(raw["synthetic"]),
        )


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Market data settings: valuation date, discounting and forward curve source."""

    valuation_date: dt.date
    discount_rate: float
    forward_curve: ForwardCurveConfig

    def __post_init__(self) -> None:
        _finite(self.discount_rate, "discount_rate")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MarketConfig:
        raw = _as_mapping(raw, "market")
        _check_keys(cls, raw, "market")
        return cls(
            valuation_date=_as_date(raw["valuation_date"], "valuation_date"),
            discount_rate=float(raw["discount_rate"]),
            forward_curve=ForwardCurveConfig.from_dict(raw["forward_curve"]),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    r"""Price model parameters: $dX_t = -\kappa X_t\,dt + \sigma\,dW_t$ (annualised)."""

    name: ModelName
    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        _require(self.name in ("one_factor", "two_factor"), f"unknown model {self.name!r}")
        _require(self.kappa > 0, "kappa must be > 0")
        _require(self.sigma >= 0, "sigma must be >= 0")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelConfig:
        raw = _as_mapping(raw, "model")
        _check_keys(cls, raw, "model")
        return cls(name=raw["name"], kappa=float(raw["kappa"]), sigma=float(raw["sigma"]))


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Monte Carlo and LSMC settings.

    Two independent path sets are used: one to fit the continuation-value regressions,
    one to evaluate the resulting policy (unbiased evaluation of a sub-optimal policy,
    hence a low-biased estimate of the storage value).
    """

    time_step: TimeStep
    n_paths_regression: int
    n_paths_valuation: int
    seed_regression: int
    seed_valuation: int
    volume_grid_size: int
    basis: BasisName
    basis_degree: int

    def __post_init__(self) -> None:
        _require(self.time_step in ("daily", "weekly"), f"unknown time_step {self.time_step!r}")
        _require(self.n_paths_regression > 0, "n_paths_regression must be > 0")
        _require(self.n_paths_valuation > 0, "n_paths_valuation must be > 0")
        _require(
            self.seed_regression != self.seed_valuation,
            "seed_regression and seed_valuation must differ (independent path sets)",
        )
        _require(self.volume_grid_size >= 2, "volume_grid_size must be >= 2")
        _require(self.basis in ("polynomial", "laguerre"), f"unknown basis {self.basis!r}")
        _require(self.basis_degree >= 1, "basis_degree must be >= 1")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MonteCarloConfig:
        raw = _as_mapping(raw, "monte_carlo")
        _check_keys(cls, raw, "monte_carlo")
        ints = {"n_paths_regression", "n_paths_valuation", "seed_regression", "seed_valuation"}
        ints |= {"volume_grid_size", "basis_degree"}
        return cls(**{k: int(v) if k in ints else v for k, v in raw.items()})


@dataclass(frozen=True, slots=True)
class Config:
    """Full run configuration."""

    contract: ContractConfig
    market: MarketConfig
    model: ModelConfig
    monte_carlo: MonteCarloConfig

    def __post_init__(self) -> None:
        _require(
            self.market.valuation_date <= self.contract.start_date,
            "valuation_date must be on or before the contract start_date",
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Config:
        raw = _as_mapping(raw, "root")
        _check_keys(cls, raw, "root")
        return cls(
            contract=ContractConfig.from_dict(raw["contract"]),
            market=MarketConfig.from_dict(raw["market"]),
            model=ModelConfig.from_dict(raw["model"]),
            monte_carlo=MonteCarloConfig.from_dict(raw["monte_carlo"]),
        )


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.from_dict(raw)
