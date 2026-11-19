"""Tests for the storage contract, intrinsic LP / MILP, volume-grid DP and rolling intrinsic."""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest

from gasstorage.config import RatchetTier, load_config
from gasstorage.data.forward_curve import ForwardCurve
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.contract import StorageContract, build_inputs
from gasstorage.storage.dp import solve_dp
from gasstorage.storage.intrinsic import solve_lp
from gasstorage.storage.rolling_intrinsic import rolling_intrinsic
from gasstorage.timegrid import decision_grid

FLAT = (RatchetTier(0.0, 1.0, 1.0),)


def _toy(rate: float = 100.0, **kw: object) -> StorageContract:
    base = {
        "working_volume": 100.0,
        "initial_inventory": 0.0,
        "terminal_inventory": 0.0,
        "terminal_penalty": None,
        "max_injection_per_day": rate,
        "max_withdrawal_per_day": rate,
        "injection_ratchets": FLAT,
        "withdrawal_ratchets": FLAT,
        "injection_cost": 0.0,
        "withdrawal_cost": 0.0,
        "fuel_loss": 0.0,
    }
    base.update(kw)
    return StorageContract(**base)  # type: ignore[arg-type]


ONES2 = np.ones(2)

# --- time grid and contract ----------------------------------------------------------------


def test_weekly_grid_covers_storage_year() -> None:
    g = decision_grid(dt.date(2027, 4, 1), dt.date(2028, 3, 31), "weekly")
    assert len(g.dates) == 53  # 366 days = 52 weeks + 2 days
    assert g.days.sum() == 366
    assert g.days[-1] == 2
    assert g.dates[1] == pd.Timestamp("2027-04-08")


def test_daily_grid_has_one_step_per_day() -> None:
    g = decision_grid(dt.date(2027, 4, 1), dt.date(2028, 3, 31), "daily")
    assert len(g.dates) == 366
    assert np.all(g.days == 1)


def test_ratchet_factors_use_half_open_tiers() -> None:
    c = StorageContract.from_config(load_config("configs/storage_default.yaml").contract)
    np.testing.assert_allclose(c.injection_factor([0.0, 0.69, 0.7, 1.0]), [1, 1, 0.6, 0.6])
    np.testing.assert_allclose(c.withdrawal_factor([0.0, 0.29, 0.3, 1.0]), [0.5, 0.5, 1, 1])
    assert c.max_injection_per_day == pytest.approx(1_000_000 / 150)


def test_cash_flow_includes_costs_fuel_and_discount() -> None:
    c = _toy(injection_cost=1.0, withdrawal_cost=2.0, fuel_loss=0.1)
    # inject 10 at 20: pays 10 * (20 * 1.1 + 1) = 230, discounted by 0.5
    # withdraw 10 at 20: receives 10 * (20 - 2) = 180, discounted by 0.5
    np.testing.assert_allclose(c.cash_flow(np.array([10.0, -10.0]), 20.0, 0.5), [-115.0, 90.0])


# --- intrinsic: hand-computed toy cases ------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "prices", "want"),
    [
        ({}, [10.0, 20.0], 1000.0),
        ({"injection_cost": 1.0, "withdrawal_cost": 1.0}, [10.0, 20.0], 800.0),
        ({"injection_cost": 1.0, "withdrawal_cost": 1.0, "fuel_loss": 0.1}, [10.0, 20.0], 700.0),
        ({}, [20.0, 10.0], 0.0),
    ],
)
@pytest.mark.parametrize("solver", ["lp", "dp"])
def test_toy_two_day_values(kw: dict, prices: list[float], want: float, solver: str) -> None:
    c = _toy(**kw)
    p = np.array(prices)
    if solver == "lp":
        res = solve_lp(c, p, ONES2, ONES2)
    else:
        res = solve_dp(c, p, ONES2, ONES2, n_levels=101)
    assert res.value == pytest.approx(want, abs=1e-6)


def _ratchet_toy() -> StorageContract:
    # Withdrawal halved below 70 % fill. From v = 100 the two selling days cannot empty the
    # storage: keeping >= 70 after day 3 caps day 3 at 30 and day 4 at 60; going below 70 caps
    # day 4 at 30 and needs >= 70 on day 3 (> rate 60). The largest sellable stock is 60 + 30.
    return _toy(
        rate=60.0,
        withdrawal_ratchets=(RatchetTier(0.0, 0.7, 0.5), RatchetTier(0.7, 1.0, 1.0)),
    )


def test_ratchets_reduce_value_on_hand_example() -> None:
    c = _ratchet_toy()
    p, ones = np.array([10.0, 10.0, 30.0, 30.0]), np.ones(4)
    assert solve_lp(c, p, ones, ones, ratchets=False).value == pytest.approx(2000.0, abs=1e-6)
    assert solve_lp(c, p, ones, ones, ratchets=True).value == pytest.approx(1800.0, abs=1e-6)
    assert solve_dp(c, p, ones, ones, n_levels=101).value == pytest.approx(1800.0, abs=1e-6)
    no_ratchet = solve_dp(c.without_ratchets(), p, ones, ones, n_levels=101).value
    assert no_ratchet == pytest.approx(2000.0, abs=1e-6)


# --- intrinsic on the real configuration ----------------------------------------------------


def _inputs(config: str, step: str = "weekly"):
    cfg = load_config(config)
    return cfg, build_inputs(cfg, step)


@pytest.fixture(scope="module")
def contango():
    return _inputs("configs/storage_synthetic.yaml")


def test_lp_solution_respects_all_constraints(contango) -> None:
    _, inp = contango
    c = inp.contract
    res = solve_lp(c, inp.prices, inp.days, inp.discount, ratchets=True)
    v = res.inventory
    tol = 1e-6 * c.working_volume
    assert v[0] == pytest.approx(c.initial_inventory)
    assert v[-1] == pytest.approx(c.terminal_inventory, abs=tol)
    assert np.all(v >= -tol) and np.all(v <= c.working_volume + tol)
    np.testing.assert_allclose(np.diff(v), res.injection - res.withdrawal, atol=tol)
    fill = v[:-1] / c.working_volume
    assert np.all(res.injection <= c.max_injection(fill * c.working_volume, inp.days) + tol)
    assert np.all(res.withdrawal <= c.max_withdrawal(fill * c.working_volume, inp.days) + tol)
    assert res.value > 0


def test_lp_equals_dp_without_ratchets(contango) -> None:
    _, inp = contango
    c = inp.contract.without_ratchets()
    lp = solve_lp(c, inp.prices, inp.days, inp.discount)
    dp = solve_dp(c, inp.prices, inp.days, inp.discount, n_levels=101)
    assert dp.value == pytest.approx(lp.value, rel=2e-3)
    assert dp.value <= lp.value + 1e-6  # the grid policy is feasible for the LP


def test_dp_converges_to_milp_with_ratchets(contango) -> None:
    # Tier ratchets make the value function discontinuous at tier boundaries: the MILP can stop
    # just below a threshold, the grid only at the level below, so the DP error is O(grid step).
    _, inp = contango
    milp = solve_lp(inp.contract, inp.prices, inp.days, inp.discount, ratchets=True).value
    dp = [
        solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=g).value
        for g in (51, 101, 201, 1001)
    ]
    assert all(a <= b + 1e-6 for a, b in itertools.pairwise(dp))  # finer grid, more value
    assert dp[-1] <= milp + 1e-6  # every grid policy is feasible for the MILP
    assert dp[-1] == pytest.approx(milp, rel=1e-3)
    lp = solve_lp(inp.contract, inp.prices, inp.days, inp.discount, ratchets=False).value
    assert milp <= lp + 1e-6


def test_dp_ends_at_terminal_inventory(contango) -> None:
    _, inp = contango
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=101)
    assert dp.inventory[-1] == pytest.approx(inp.contract.terminal_inventory, abs=1e-6)


def test_backwardated_market_curve_has_zero_intrinsic_value() -> None:
    _, inp = _inputs("configs/storage_default.yaml")
    assert np.all(np.diff(inp.prices) <= 0)
    res = solve_lp(inp.contract, inp.prices, inp.days, inp.discount, ratchets=True)
    assert res.value == pytest.approx(0.0, abs=1e-6)


def test_flat_curve_with_costs_is_worthless(contango) -> None:
    _, inp = contango
    flat = np.full_like(inp.prices, 30.0)
    assert solve_lp(inp.contract, flat, inp.days, inp.discount).value == pytest.approx(0, abs=1e-6)
    assert solve_dp(inp.contract, flat, inp.days, inp.discount).value == pytest.approx(0, abs=1e-6)


def test_value_increases_with_capacity_and_rates(contango) -> None:
    _, inp = contango
    c = inp.contract

    def value(contract: StorageContract) -> float:
        return solve_lp(contract, inp.prices, inp.days, inp.discount).value

    base = value(c)
    bigger = dataclasses.replace(
        c,
        working_volume=1.5 * c.working_volume,
        max_injection_per_day=1.5 * c.max_injection_per_day,
        max_withdrawal_per_day=1.5 * c.max_withdrawal_per_day,
    )
    faster = dataclasses.replace(c, max_injection_per_day=2 * c.max_injection_per_day)
    assert value(bigger) > base
    assert value(faster) > base


def test_soft_terminal_constraint_is_worth_at_least_the_hard_one(contango) -> None:
    _, inp = contango
    soft = dataclasses.replace(inp.contract, terminal_penalty=5.0)
    hard = solve_lp(inp.contract, inp.prices, inp.days, inp.discount).value
    assert solve_lp(soft, inp.prices, inp.days, inp.discount).value >= hard - 1e-6


# --- rolling intrinsic ------------------------------------------------------------------------


def test_rolling_intrinsic_equals_intrinsic_without_volatility(contango) -> None:
    _, inp = contango
    model = OneFactorModel(kappa=4.0, sigma=0.0)
    ri = rolling_intrinsic(inp, model, n_paths=20, rng=np.random.default_rng(0), n_levels=51)
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=51)
    assert ri.value == pytest.approx(dp.value, rel=1e-9)
    assert ri.se == pytest.approx(0.0, abs=1e-6)


def test_rolling_intrinsic_is_at_least_intrinsic(contango) -> None:
    cfg, inp = contango
    model = OneFactorModel.from_config(cfg.model)
    ri = rolling_intrinsic(inp, model, n_paths=400, rng=np.random.default_rng(1), n_levels=51)
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=51)
    assert ri.value >= dp.value - 3 * ri.se
    assert ri.inventory.shape == (len(inp.prices) + 1, 400)
    assert np.allclose(ri.inventory[-1], inp.contract.terminal_inventory, atol=1e-6)


def test_build_inputs_uses_forward_curve_and_discounting() -> None:
    cfg, inp = _inputs("configs/storage_default.yaml")
    curve = ForwardCurve.from_config(
        cfg.market.forward_curve, cfg.contract.start_date, cfg.contract.end_date
    )
    np.testing.assert_allclose(inp.prices, curve.daily(inp.dates))
    # first step on 2027-04-01: tau = 135 / 365 from the 2026-11-17 valuation date
    assert inp.times[0] == pytest.approx(135 / 365)
    assert inp.discount[0] == pytest.approx(np.exp(-0.03 * 135 / 365))
