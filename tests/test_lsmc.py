"""Tests for the regression basis and the LSMC storage valuation."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from gasstorage.config import load_config
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.basis import laguerre_basis, polynomial_basis, state_basis
from gasstorage.storage.contract import build_inputs
from gasstorage.storage.dp import solve_dp
from gasstorage.storage.lsmc import lsmc_value

# --- basis ----------------------------------------------------------------------------------


def test_polynomial_basis_columns() -> None:
    phi = polynomial_basis(np.array([2.0, -1.0]), degree=3)
    np.testing.assert_allclose(phi, [[1, 2, 4, 8], [1, -1, 1, -1]])


def test_weighted_laguerre_basis_columns() -> None:
    # L0 = 1, L1 = 1 - x, L2 = (x^2 - 4x + 2) / 2, each weighted by exp(-x / 2)
    phi = laguerre_basis(np.array([1.0]), degree=2)
    w = np.exp(-0.5)
    np.testing.assert_allclose(phi, [[w, 0.0, -0.5 * w]])


def test_state_basis_normalises_by_model_standard_deviation() -> None:
    model = OneFactorModel(kappa=2.0, sigma=0.5)
    t = 0.4
    sd = np.sqrt(model.variance(t))
    phi = state_basis(np.array([sd, -2 * sd]), model, t, "polynomial", degree=2)
    np.testing.assert_allclose(phi, [[1, 1, 1], [1, -2, 4]])


def test_state_basis_handles_zero_variance() -> None:
    model = OneFactorModel(kappa=2.0, sigma=0.0)
    phi = state_basis(np.zeros(3), model, 0.4, "polynomial", degree=2)
    assert np.all(np.isfinite(phi))


# --- LSMC -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contango():
    cfg = load_config("configs/storage_synthetic.yaml")
    return cfg, build_inputs(cfg, "weekly")


@pytest.fixture(scope="module")
def market():
    cfg = load_config("configs/storage_default.yaml")
    return cfg, build_inputs(cfg, "weekly")


def _run(inp, model, n=1000, seeds=(1, 2), **kw):
    return lsmc_value(
        inp,
        model,
        n_paths_regression=n,
        n_paths_valuation=n,
        rng_regression=np.random.default_rng(seeds[0]),
        rng_valuation=np.random.default_rng(seeds[1]),
        **kw,
    )


def test_zero_volatility_lsmc_equals_intrinsic(contango) -> None:
    _, inp = contango
    res = _run(inp, OneFactorModel(kappa=4.0, sigma=0.0), n=50, n_levels=51)
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=51)
    assert res.value == pytest.approx(dp.value, rel=1e-9)
    assert res.se == pytest.approx(0.0, abs=1e-6)


def test_small_volatility_lsmc_is_close_to_intrinsic(contango) -> None:
    _, inp = contango
    res = _run(inp, OneFactorModel(kappa=4.0, sigma=0.01), n=500, n_levels=51)
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=51).value
    assert abs(res.value - dp) / dp < 0.02


@pytest.mark.parametrize("case", ["contango", "market"])
def test_lsmc_is_at_least_intrinsic(case, request) -> None:
    cfg, inp = request.getfixturevalue(case)
    res = _run(inp, OneFactorModel.from_config(cfg.model), n=2000, n_levels=51)
    dp = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=51).value
    assert res.value >= dp - 3 * res.se
    assert np.allclose(res.inventory[-1], inp.contract.terminal_inventory, atol=1e-6)
    vol = inp.contract.working_volume
    assert np.all(res.inventory >= -1e-6) and np.all(res.inventory <= vol + 1e-6)


def test_lsmc_value_increases_with_volatility(contango) -> None:
    cfg, inp = contango
    values = [
        _run(inp, OneFactorModel(kappa=cfg.model.kappa, sigma=s), n=2000, n_levels=51)
        for s in (0.2, 0.5, 0.8)
    ]
    for lo, hi in itertools.pairwise(values):
        assert hi.value - lo.value > 3 * np.hypot(hi.se, lo.se)


def test_lsmc_converges_in_number_of_paths(contango) -> None:
    cfg, inp = contango
    model = OneFactorModel.from_config(cfg.model)
    small = _run(inp, model, n=1000, seeds=(3, 4), n_levels=51)
    large = _run(inp, model, n=4000, seeds=(5, 6), n_levels=51)
    assert abs(small.value - large.value) < 3 * np.hypot(small.se, large.se) + 0.01 * large.value
    assert large.se < small.se


@pytest.mark.slow
def test_lsmc_converges_in_grid_and_basis(contango) -> None:
    cfg, inp = contango
    model = OneFactorModel.from_config(cfg.model)
    v51 = _run(inp, model, n=2000, n_levels=51).value
    v101 = _run(inp, model, n=2000, n_levels=101).value
    assert v101 > v51  # coarser grid, more discretisation loss (same paths)
    assert abs(v101 - v51) / v101 < 0.05
    d2 = _run(inp, model, n=2000, n_levels=51, degree=2).value
    d4 = _run(inp, model, n=2000, n_levels=51, degree=4).value
    assert abs(d4 - d2) / d4 < 0.02
    lag = _run(inp, model, n=2000, n_levels=51, basis="laguerre", degree=3).value
    assert abs(lag - v51) / v51 < 0.02


def test_in_sample_estimate_is_reported_and_seeds_are_independent(contango) -> None:
    cfg, inp = contango
    res = _run(inp, OneFactorModel.from_config(cfg.model), n=1000, n_levels=51)
    # In-sample and out-of-sample estimates of the same policy agree to Monte Carlo error; a
    # look-ahead bug (deciding on realised instead of regressed continuation values) inflates the
    # in-sample estimate by tens of standard errors.
    assert abs(res.in_sample_value - res.value) < 5 * np.sqrt(2) * res.se
    with pytest.raises(ValueError, match="independent"):
        lsmc_value(
            inp,
            OneFactorModel.from_config(cfg.model),
            n_paths_regression=10,
            n_paths_valuation=10,
            rng_regression=np.random.default_rng(1),
            rng_valuation=np.random.default_rng(1),
        )


def test_soft_terminal_condition_is_supported(contango) -> None:
    cfg, inp = contango
    soft = dataclasses.replace(
        inp, contract=dataclasses.replace(inp.contract, terminal_penalty=1.0)
    )
    res = _run(soft, OneFactorModel.from_config(cfg.model), n=500, n_levels=51)
    assert np.isfinite(res.value)


def test_policy_map_returns_feasible_moves(contango) -> None:
    from gasstorage.storage.lsmc import policy_map

    cfg, inp = contango
    model = OneFactorModel.from_config(cfg.model)
    res = _run(inp, model, n=500, n_levels=51)
    k = 10
    x_values = np.linspace(-0.5, 0.5, 7)
    inventories = np.linspace(0, inp.contract.working_volume, 5)
    dv = policy_map(res, inp, model, k, x_values, inventories)
    assert dv.shape == (7, 5)
    c = inp.contract
    assert np.all(dv <= c.max_injection(inventories, inp.days[k])[None, :] + 1e-6)
    assert np.all(-dv <= c.max_withdrawal(inventories, inp.days[k])[None, :] + 1e-6)
    assert np.all(inventories[None, :] + dv >= -1e-6)
    assert np.all(inventories[None, :] + dv <= c.working_volume + 1e-6)
