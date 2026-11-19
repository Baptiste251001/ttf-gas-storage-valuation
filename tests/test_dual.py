"""Tests for the grid-restricted problem and its dual (information-relaxation) upper bound."""

from __future__ import annotations

import numpy as np
import pytest

from gasstorage.config import load_config
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.contract import build_inputs
from gasstorage.storage.dp import GridMoves, MoveSearch, volume_levels
from gasstorage.storage.dual import conditional_basis_mean, dual_upper_bound, normal_raw_moments
from gasstorage.storage.lsmc import lsmc_value
from tests.conftest import skip_without_market_data


def test_normal_raw_moments() -> None:
    # E[(1 + 2Z)^j], j = 0..3: 1, 1, 1 + 4, 1 + 3 * 1 * 4
    np.testing.assert_allclose(normal_raw_moments(np.array([1.0]), 2.0, 3)[0], [1, 1, 5, 13])


def test_conditional_basis_mean_matches_simulation() -> None:
    model = OneFactorModel(kappa=3.0, sigma=0.7)
    t0, t1 = 0.4, 0.5
    x0 = np.array([-0.3, 0.2])
    want = conditional_basis_mean(model, x0, t0, t1, degree=3)
    rng = np.random.default_rng(0)
    a = np.exp(-3.0 * 0.1)
    s = np.sqrt(model.variance(t0, t1))
    for i, x in enumerate(x0):
        x1 = a * x + s * rng.standard_normal(400_000)
        phi = model.basis(x1, t1, "polynomial", 3)
        se = phi.std(axis=0) / np.sqrt(len(x1))
        assert np.all(np.abs(phi.mean(axis=0) - want[i]) < 4 * se + 1e-12)


def test_grid_only_moves_land_on_grid_levels() -> None:
    inp = build_inputs(load_config("configs/storage_synthetic.yaml"), "weekly")
    levels = volume_levels(inp.contract, 51)
    search = MoveSearch.build(inp.contract, levels, 7.0, extremes=False)
    gm = GridMoves.build(inp.contract, levels, 7.0, search)
    targets = levels[:, None] + gm.dv
    on_grid = np.isclose(targets / search.delta, np.round(targets / search.delta))
    assert np.all(on_grid[gm.feasible])
    # grid targets are exact: weight 0, or 1 for the top level (stored as i0 = G-2, w = 1)
    w = gm.w[gm.feasible]
    assert np.all(np.isclose(w, 0.0) | np.isclose(w, 1.0))


@pytest.fixture(scope="module")
def contango():
    cfg = load_config("configs/storage_synthetic.yaml")
    return cfg, build_inputs(cfg, "weekly")


def _bounds(inp, model, n=1000, levels=101):
    # The grid step must not exceed the smallest ratcheted weekly move (35 GWh withdrawal below
    # 30 % fill): with 26 levels (40 GWh steps) the grid-restricted storage could never be emptied
    # once below 30 %, and its value would be 0.
    lb = lsmc_value(
        inp,
        model,
        n,
        n,
        np.random.default_rng(1),
        np.random.default_rng(2),
        n_levels=levels,
        moves="grid",
    )
    ub = dual_upper_bound(inp, model, lb.policy, n, np.random.default_rng(3))
    return lb, ub


def test_zero_volatility_bounds_coincide(contango) -> None:
    _, inp = contango
    lb, ub = _bounds(inp, OneFactorModel(kappa=4.0, sigma=0.0), n=20)
    assert ub.value == pytest.approx(lb.value, rel=1e-9)
    assert ub.se == pytest.approx(0.0, abs=1e-6)


def test_upper_bound_dominates_lower_bound(contango) -> None:
    cfg, inp = contango
    lb, ub = _bounds(inp, OneFactorModel.from_config(cfg.model))
    assert ub.value >= lb.value - 3 * np.hypot(ub.se, lb.se)
    assert ub.value - lb.value < 0.1 * lb.value  # the regression martingale is informative


def test_upper_bound_on_market_curve() -> None:
    skip_without_market_data()
    cfg = load_config("configs/storage_default.yaml")
    inp = build_inputs(cfg, "weekly")
    lb, ub = _bounds(inp, OneFactorModel.from_config(cfg.model))
    assert ub.value >= lb.value - 3 * np.hypot(ub.se, lb.se)


def test_penalties_have_zero_mean(contango) -> None:
    cfg, inp = contango
    model = OneFactorModel.from_config(cfg.model)
    _, ub = _bounds(inp, model, n=2000)
    pen = ub.mean_penalty  # (K, G): average penalty per step and target level
    se = ub.penalty_se
    finite = np.isfinite(se) & (se > 0)
    assert np.mean(np.abs(pen[finite]) < 4 * se[finite]) > 0.95
