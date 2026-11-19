"""Tests for YAML configuration loading and validation."""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from gasstorage.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "storage_default.yaml"


def _valid_raw() -> dict[str, Any]:
    """A minimal, hand-written valid configuration (independent of the shipped YAML)."""
    return {
        "contract": {
            "start_date": "2026-04-01",
            "end_date": "2027-03-31",
            "working_volume_mwh": 1_000_000,
            "initial_inventory_mwh": 0,
            "terminal_inventory_mwh": 0,
            "terminal_penalty_eur_mwh": None,
            "injection_days_to_fill": 150,
            "withdrawal_days_to_empty": 100,
            "injection_ratchets": [
                {"lower": 0.0, "upper": 0.7, "factor": 1.0},
                {"lower": 0.7, "upper": 1.0, "factor": 0.6},
            ],
            "withdrawal_ratchets": [
                {"lower": 0.0, "upper": 0.3, "factor": 0.5},
                {"lower": 0.3, "upper": 1.0, "factor": 1.0},
            ],
            "injection_cost_eur_mwh": 0.5,
            "withdrawal_cost_eur_mwh": 0.5,
            "fuel_loss": 0.0,
        },
        "market": {
            "valuation_date": "2026-03-31",
            "discount_rate": 0.03,
            "forward_curve": {
                "source": "synthetic",
                "csv_path": None,
                "synthetic": {
                    "base_level_eur_mwh": 30.0,
                    "winter_premium_eur_mwh": 5.0,
                    "contango_eur_mwh_per_year": 1.0,
                },
            },
        },
        "model": {"name": "one_factor", "kappa": 5.0, "sigma": 0.6},
        "monte_carlo": {
            "time_step": "weekly",
            "n_paths_regression": 10_000,
            "n_paths_valuation": 10_000,
            "seed_regression": 1,
            "seed_valuation": 2,
            "volume_grid_size": 101,
            "basis": "polynomial",
            "basis_degree": 3,
        },
    }


def _with(path: str, value: Any) -> dict[str, Any]:
    """Return a copy of the valid config with the dotted ``path`` set to ``value``."""
    raw = _valid_raw()
    node = raw
    *parents, leaf = path.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = copy.deepcopy(value)
    return raw


def test_shipped_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert isinstance(cfg, Config)


def test_max_rates_are_derived_from_days_to_fill() -> None:
    cfg = Config.from_dict(_valid_raw())
    assert cfg.contract.max_injection_mwh_per_day == pytest.approx(1_000_000 / 150)
    assert cfg.contract.max_withdrawal_mwh_per_day == pytest.approx(10_000.0)


def test_dates_are_parsed_to_date_objects() -> None:
    cfg = Config.from_dict(_valid_raw())
    assert cfg.contract.start_date == dt.date(2026, 4, 1)
    assert cfg.contract.end_date == dt.date(2027, 3, 31)
    assert cfg.market.valuation_date == dt.date(2026, 3, 31)


def test_ratchets_are_parsed_into_tiers() -> None:
    cfg = Config.from_dict(_valid_raw())
    tiers = cfg.contract.withdrawal_ratchets
    assert [(t.lower, t.upper, t.factor) for t in tiers] == [(0.0, 0.3, 0.5), (0.3, 1.0, 1.0)]


def test_config_is_immutable() -> None:
    cfg = Config.from_dict(_valid_raw())
    with pytest.raises(AttributeError):
        cfg.contract.working_volume_mwh = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("contract.end_date", "2026-03-01", "end_date"),
        ("contract.working_volume_mwh", 0, "working_volume_mwh"),
        ("contract.initial_inventory_mwh", 1_000_001, "initial_inventory_mwh"),
        ("contract.terminal_inventory_mwh", -1, "terminal_inventory_mwh"),
        ("contract.terminal_penalty_eur_mwh", -5.0, "terminal_penalty"),
        ("contract.injection_days_to_fill", 0, "injection_days_to_fill"),
        ("contract.fuel_loss", 1.0, "fuel_loss"),
        ("contract.injection_cost_eur_mwh", -0.1, "injection_cost"),
        # ratchet tiers: gap, not starting at 0, not ending at 1, factor out of (0, 1]
        (
            "contract.withdrawal_ratchets",
            [
                {"lower": 0.0, "upper": 0.3, "factor": 0.5},
                {"lower": 0.4, "upper": 1.0, "factor": 1},
            ],
            "contiguous",
        ),
        ("contract.withdrawal_ratchets", [{"lower": 0.1, "upper": 1.0, "factor": 1.0}], "cover"),
        ("contract.withdrawal_ratchets", [{"lower": 0.0, "upper": 0.9, "factor": 1.0}], "cover"),
        ("contract.injection_ratchets", [{"lower": 0.0, "upper": 1.0, "factor": 0.0}], "factor"),
        ("contract.injection_ratchets", [{"lower": 0.0, "upper": 1.0, "factor": 1.2}], "factor"),
        ("contract.injection_ratchets", [], "at least one"),
        ("market.forward_curve.source", "bloomberg", "source"),
        ("market.forward_curve.source", "csv", "csv_path"),
        ("model.name", "heston", "model"),
        ("model.kappa", 0.0, "kappa"),
        ("model.sigma", -0.1, "sigma"),
        ("monte_carlo.time_step", "hourly", "time_step"),
        ("monte_carlo.n_paths_valuation", 0, "n_paths"),
        ("monte_carlo.seed_valuation", 1, "seed"),
        ("monte_carlo.volume_grid_size", 1, "volume_grid_size"),
        ("monte_carlo.basis", "chebyshev", "basis"),
        ("monte_carlo.basis_degree", 0, "basis_degree"),
    ],
)
def test_invalid_values_are_rejected(path: str, value: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Config.from_dict(_with(path, value))


def test_zero_sigma_is_allowed_for_deterministic_limit() -> None:
    # sigma -> 0 is a required LSMC test case (LSMC value -> intrinsic value).
    cfg = Config.from_dict(_with("model.sigma", 0.0))
    assert cfg.model.sigma == 0.0


def test_unknown_key_is_rejected() -> None:
    raw = _valid_raw()
    raw["contract"]["max_rate"] = 1.0
    with pytest.raises(ValueError, match="max_rate"):
        Config.from_dict(raw)


def test_missing_key_is_rejected() -> None:
    raw = _valid_raw()
    del raw["contract"]["fuel_loss"]
    with pytest.raises(ValueError, match="fuel_loss"):
        Config.from_dict(raw)


def test_valuation_date_after_contract_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="valuation_date"):
        Config.from_dict(_with("market.valuation_date", "2026-04-02"))


def test_valuation_date_on_contract_start_is_accepted() -> None:
    cfg = Config.from_dict(_with("market.valuation_date", "2026-04-01"))
    assert cfg.market.valuation_date == cfg.contract.start_date


def test_csv_source_with_path_is_accepted() -> None:
    raw = _with("market.forward_curve.source", "csv")
    raw["market"]["forward_curve"]["csv_path"] = "data/raw/curve.csv"
    cfg = Config.from_dict(raw)
    assert cfg.market.forward_curve.csv_path == Path("data/raw/curve.csv")
