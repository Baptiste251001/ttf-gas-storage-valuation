"""Tests for bump-and-revalue greeks with common random numbers and the delta hedge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gasstorage.config import load_config
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.risk.delta_hedge import delta_hedge, pnl_statistics
from gasstorage.risk.greeks import bump_month, bump_parallel, lsmc_greeks
from gasstorage.storage.contract import build_inputs
from gasstorage.storage.dp import solve_dp
from gasstorage.storage.lsmc import lsmc_value
from tests.conftest import skip_without_market_data


@pytest.fixture(scope="module")
def contango():
    cfg = load_config("configs/storage_synthetic.yaml")
    return cfg, build_inputs(cfg, "weekly")


@pytest.fixture(scope="module")
def market():
    skip_without_market_data()
    cfg = load_config("configs/storage_default.yaml")
    return cfg, build_inputs(cfg, "weekly")


# --- bumps ----------------------------------------------------------------------------------


def test_bump_month_scales_only_that_month(contango) -> None:
    _, inp = contango
    bumped = bump_month(inp, pd.Period("2027-07", "M"), 0.01)
    in_july = inp.dates.to_period("M") == pd.Period("2027-07", "M")
    np.testing.assert_allclose(bumped.prices[in_july], 1.01 * inp.prices[in_july])
    np.testing.assert_allclose(bumped.prices[~in_july], inp.prices[~in_july])


def test_bump_parallel_shifts_every_price(contango) -> None:
    _, inp = contango
    np.testing.assert_allclose(bump_parallel(inp, 0.5).prices, inp.prices + 0.5)


# --- greeks ---------------------------------------------------------------------------------


def test_zero_volatility_monthly_delta_is_discounted_net_volume(contango) -> None:
    _, inp = contango
    model = OneFactorModel(kappa=4.0, sigma=0.0)
    months = [pd.Period("2027-06", "M"), pd.Period("2028-01", "M")]
    # Without volatility the value is piecewise linear in the curve: the bump must be smaller than
    # the distance to the next change of optimal schedule (a 1 % January bump crosses one).
    g = lsmc_greeks(inp, model, n_paths=20, seeds=(1, 2), n_levels=51, months=months, rel_bump=1e-3)
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=51)
    loss = inp.contract.fuel_loss
    net = np.where(dp.moves > 0, -(1 + loss) * dp.moves, -dp.moves)
    for m in months:
        mask = inp.dates.to_period("M") == m
        want = float(np.sum(inp.discount[mask] * net[mask]))
        assert g.deltas.loc[m, "delta_mwh"] == pytest.approx(want, rel=1e-6, abs=1e-3)
    # June: injection (buyer, negative delta); January: withdrawal (seller, positive delta)
    assert g.deltas.loc[months[0], "delta_mwh"] < 0 < g.deltas.loc[months[1], "delta_mwh"]


def test_greeks_are_stable_across_seeds_with_crn(market) -> None:
    cfg, inp = market
    model = OneFactorModel.from_config(cfg.model)
    kw = {"n_paths": 1500, "n_levels": 51, "months": []}
    a = lsmc_greeks(inp, model, seeds=(11, 12), **kw)
    b = lsmc_greeks(inp, model, seeds=(13, 14), **kw)
    for name in ("parallel_delta", "vega"):
        ga, gb = getattr(a, name), getattr(b, name)
        assert abs(ga.value - gb.value) < 3 * np.hypot(ga.se, gb.se) + 0.05 * abs(ga.value)
    assert a.vega.value > 3 * a.vega.se  # optionality is worth more with more volatility


def test_crn_makes_greek_errors_small(market) -> None:
    cfg, inp = market
    model = OneFactorModel.from_config(cfg.model)
    g = lsmc_greeks(inp, model, n_paths=1500, seeds=(11, 12), n_levels=51, months=[])
    # the standard error of the value itself is several times larger than a 0.5 EUR/MWh
    # parallel-shift P&L error when the same paths are reused
    assert g.parallel_delta.se * 0.5 < 0.2 * g.base_se


@pytest.mark.slow
def test_parallel_delta_matches_sum_of_monthly_deltas(contango) -> None:
    cfg, inp = contango
    model = OneFactorModel.from_config(cfg.model)
    g = lsmc_greeks(inp, model, n_paths=1500, seeds=(21, 22), n_levels=51)
    total = g.deltas["delta_mwh"].sum()
    se = np.sqrt((g.deltas["se"] ** 2).sum() + g.parallel_delta.se**2)
    assert abs(total - g.parallel_delta.value) < 3 * se + 0.05 * abs(g.parallel_delta.value)


# --- delta hedge ----------------------------------------------------------------------------


def _policy(cfg, inp, model, n=1500):
    return lsmc_value(
        inp, model, n, 10, np.random.default_rng(31), np.random.default_rng(32), n_levels=51
    ).policy


def test_hedge_is_neutral_without_volatility(contango) -> None:
    cfg, inp = contango
    model = OneFactorModel(kappa=4.0, sigma=0.0)
    res = delta_hedge(
        inp,
        model,
        _policy(cfg, inp, model, 50),
        50,
        50,
        np.random.default_rng(1),
        np.random.default_rng(2),
    )
    np.testing.assert_allclose(res.hedge_pnl, 0.0, atol=1e-6)
    np.testing.assert_allclose(res.hedged, res.unhedged, atol=1e-6)


def test_delta_hedge_reduces_pnl_dispersion(market) -> None:
    cfg, inp = market
    model = OneFactorModel.from_config(cfg.model)
    res = delta_hedge(
        inp,
        model,
        _policy(cfg, inp, model),
        1500,
        1500,
        np.random.default_rng(41),
        np.random.default_rng(42),
        rebalance="weekly",
    )
    assert res.hedged.std() < 0.5 * res.unhedged.std()
    # forwards are martingales: hedging does not change the mean beyond Monte Carlo error
    diff = res.hedged - res.unhedged
    assert abs(diff.mean()) < 3 * diff.std(ddof=1) / np.sqrt(diff.size)


def test_transaction_costs_lower_the_hedged_mean(market) -> None:
    cfg, inp = market
    model = OneFactorModel.from_config(cfg.model)
    pol = _policy(cfg, inp, model)
    kw = {"rebalance": "monthly"}
    free = delta_hedge(
        inp, model, pol, 800, 800, np.random.default_rng(5), np.random.default_rng(6), **kw
    )
    paid = delta_hedge(
        inp,
        model,
        pol,
        800,
        800,
        np.random.default_rng(5),
        np.random.default_rng(6),
        bid_ask=0.2,
        **kw,
    )
    assert np.all(paid.costs >= 0)
    np.testing.assert_allclose(paid.hedged, free.hedged - paid.costs, atol=1e-6)
    assert paid.costs.mean() > 0


def test_initial_hedge_delta_matches_bump_delta(market) -> None:
    cfg, inp = market
    model = OneFactorModel.from_config(cfg.model)
    month = pd.Period("2027-04", "M")
    g = lsmc_greeks(inp, model, n_paths=2000, seeds=(51, 52), n_levels=51, months=[month])
    pol = lsmc_value(
        inp, model, 2000, 10, np.random.default_rng(51), np.random.default_rng(53), n_levels=51
    ).policy
    res = delta_hedge(
        inp, model, pol, 2000, 10, np.random.default_rng(54), np.random.default_rng(55)
    )
    k = int(np.argmax(inp.dates.to_period("M") == month))
    envelope = res.initial_delta.loc[month] * inp.discount[k]  # undiscounted MWh -> dV/dF
    bump = g.deltas.loc[month, "delta_mwh"]
    assert envelope == pytest.approx(bump, rel=0.15)


def test_pnl_statistics_columns() -> None:
    pnl = {"a": np.array([-2.0, 0.0, 1.0, 3.0] * 25)}
    stats = pnl_statistics(pnl)
    assert stats.loc["a", "mean"] == pytest.approx(0.5)
    # 5 % quantile of the sample is -2: VaR is its negative, ES the mean below it
    assert stats.loc["a", "VaR 95 %"] == pytest.approx(2.0)
    assert stats.loc["a", "ES 95 %"] == pytest.approx(2.0)
