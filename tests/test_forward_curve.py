"""Tests for the forward curve object: CSV loader, synthetic generator, daily interpolation."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gasstorage.config import ForwardCurveConfig, SyntheticCurveConfig
from gasstorage.data.forward_curve import ForwardCurve

SYN = SyntheticCurveConfig(
    base_level_eur_mwh=30.0, winter_premium_eur_mwh=5.0, contango_eur_mwh_per_year=1.0
)


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> Path:
    pd.DataFrame(rows, columns=["delivery_month", "price"]).to_csv(path, index=False)
    return path


# --- synthetic generator -----------------------------------------------------------------


def test_synthetic_curve_has_one_price_per_month() -> None:
    curve = ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, SYN)
    assert len(curve.months) == 12
    assert curve.months[0] == pd.Period("2026-04", "M")
    assert curve.months[-1] == pd.Period("2027-03", "M")
    assert curve.prices.shape == (12,)


def test_synthetic_curve_values_match_hand_computation() -> None:
    # F_m = 30 + 5 cos(2 pi (month - 1) / 12) + 1 * k / 12, k = months since start
    curve = ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, SYN)
    p = dict(zip(curve.months.strftime("%Y-%m"), curve.prices, strict=True))
    assert p["2026-04"] == pytest.approx(30.0)
    assert p["2026-07"] == pytest.approx(25.25)
    assert p["2026-10"] == pytest.approx(30.5)
    assert p["2027-01"] == pytest.approx(35.75)


def test_synthetic_curve_is_labelled_synthetic() -> None:
    curve = ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, SYN)
    assert curve.is_synthetic
    assert "synthetic" in curve.source.lower()


def test_synthetic_curve_rejects_non_positive_prices() -> None:
    bad = SyntheticCurveConfig(
        base_level_eur_mwh=3.0, winter_premium_eur_mwh=5.0, contango_eur_mwh_per_year=0.0
    )
    with pytest.raises(ValueError, match="positive"):
        ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, bad)


# --- CSV loader ----------------------------------------------------------------------------


def test_csv_loader_reads_months_and_prices(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "c.csv", [("2027-04", 31.2), ("2027-05", 30.8), ("2027-06", 30.9)])
    curve = ForwardCurve.from_csv(path)
    assert list(curve.months.strftime("%Y-%m")) == ["2027-04", "2027-05", "2027-06"]
    np.testing.assert_allclose(curve.prices, [31.2, 30.8, 30.9])
    assert not curve.is_synthetic


def test_csv_loader_sorts_unordered_months(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "c.csv", [("2027-05", 2.0), ("2027-04", 1.0)])
    curve = ForwardCurve.from_csv(path)
    np.testing.assert_allclose(curve.prices, [1.0, 2.0])


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([("2027-04", 31.0), ("2027-06", 30.0)], "contiguous"),
        ([("2027-04", 31.0), ("2027-04", 30.0)], "duplicate"),
        ([("2027-04", 31.0), ("2027-05", -1.0)], "positive"),
        ([("2027-04", 31.0), ("2027-05", float("nan"))], "positive"),
    ],
)
def test_csv_loader_rejects_bad_curves(
    tmp_path: Path, rows: list[tuple[str, float]], match: str
) -> None:
    path = _write_csv(tmp_path / "c.csv", rows)
    with pytest.raises(ValueError, match=match):
        ForwardCurve.from_csv(path)


def test_csv_loader_rejects_wrong_columns(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    pd.DataFrame({"month": ["2027-04"], "settle": [30.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="delivery_month"):
        ForwardCurve.from_csv(path)


# --- daily interpolation -------------------------------------------------------------------


def test_daily_prices_are_constant_within_each_month() -> None:
    curve = ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, SYN)
    april = pd.date_range("2026-04-01", "2026-04-30", freq="D")
    np.testing.assert_allclose(curve.daily(april), 30.0)
    assert curve.daily(pd.DatetimeIndex(["2027-01-15"]))[0] == pytest.approx(35.75)


def test_daily_prices_jump_at_month_boundary() -> None:
    curve = ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, SYN)
    p = curve.daily(pd.DatetimeIndex(["2026-06-30", "2026-07-01"]))
    assert p[0] != pytest.approx(p[1])
    assert p[1] == pytest.approx(25.25)


def test_daily_prices_outside_curve_raise() -> None:
    curve = ForwardCurve.synthetic(pd.Period("2026-04", "M"), 12, SYN)
    with pytest.raises(ValueError, match="outside"):
        curve.daily(pd.DatetimeIndex(["2027-04-01"]))


# --- construction from config --------------------------------------------------------------


def test_from_config_synthetic_covers_contract_period() -> None:
    cfg = ForwardCurveConfig(source="synthetic", csv_path=None, synthetic=SYN)
    curve = ForwardCurve.from_config(cfg, dt.date(2026, 4, 1), dt.date(2027, 3, 31))
    assert curve.months[0] == pd.Period("2026-04", "M")
    assert curve.months[-1] == pd.Period("2027-03", "M")


def test_from_config_csv_must_cover_contract_period(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "c.csv", [("2026-04", 30.0), ("2026-05", 30.0)])
    cfg = ForwardCurveConfig(source="csv", csv_path=path, synthetic=SYN)
    with pytest.raises(ValueError, match="cover"):
        ForwardCurve.from_config(cfg, dt.date(2026, 4, 1), dt.date(2027, 3, 31))
