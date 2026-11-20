"""Tests for the valuation snapshot used to compare runs on different curve dates."""

from __future__ import annotations

import numpy as np
import pytest

from gasstorage.config import load_config
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.report import curve_summary, valuation_snapshot
from gasstorage.storage.contract import build_inputs
from gasstorage.storage.intrinsic import solve_lp


def test_curve_summary_on_synthetic_curve() -> None:
    inp = build_inputs(load_config("configs/storage_synthetic.yaml"), "daily")
    s = curve_summary(inp)
    # synthetic contango curve: winter (Oct-Mar) above summer (Apr-Sep), not decreasing
    assert s["winter_minus_summer_eur_mwh"] > 0
    assert s["strictly_decreasing"] is False
    days = inp.days
    summer = inp.dates.month.isin([4, 5, 6, 7, 8, 9])
    want = float(np.sum(inp.prices[summer] * days[summer]) / days[summer].sum())
    assert s["summer_avg_eur_mwh"] == pytest.approx(want)


def test_valuation_snapshot_contents() -> None:
    cfg = load_config("configs/storage_synthetic.yaml")
    models = {"1F test": OneFactorModel(4.0, 0.5)}
    snap = valuation_snapshot(cfg, models, n_paths=300, n_levels=101, dual_for=["1F test"])
    inp = build_inputs(cfg, "weekly")
    want = solve_lp(inp.contract, inp.prices, inp.days, inp.discount, ratchets=True).value
    assert snap["intrinsic_milp_eur"] == pytest.approx(want)
    m = snap["models"]["1F test"]
    assert m["lsmc_eur"] >= snap["intrinsic_same_grid_eur"] - 3 * m["lsmc_se_eur"]
    assert m["dual_upper_eur"] >= m["grid_lower_eur"] - 3 * np.hypot(
        m["dual_se_eur"], m["grid_lower_se_eur"]
    )
    assert snap["valuation_date"] == str(cfg.market.valuation_date)
