"""Tests for the Schwartz-Smith two-factor model and its use in the valuation engine."""

from __future__ import annotations

import numpy as np
import pytest

from gasstorage.config import load_config
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.models.two_factor import TwoFactorModel
from gasstorage.storage.contract import build_inputs
from gasstorage.storage.dp import solve_dp
from gasstorage.storage.lsmc import lsmc_value
from gasstorage.storage.rolling_intrinsic import rolling_intrinsic

M2 = TwoFactorModel(kappa=1.5, sigma_chi=0.6, sigma_xi=0.25, rho=-0.3)


def test_total_variance_formula() -> None:
    # V(t,T) = s_chi^2 (1 - e^{-2k tau}) / (2k) + s_xi^2 tau
    #          + 2 rho s_chi s_xi (1 - e^{-k tau}) / k
    tau = 0.8
    want = (
        0.36 * (1 - np.exp(-3.0 * tau)) / 3.0
        + 0.0625 * tau
        + 2 * -0.3 * 0.6 * 0.25 * (1 - np.exp(-1.5 * tau)) / 1.5
    )
    assert M2.variance(0.2, 1.0) == pytest.approx(want)
    assert M2.variance(tau) == pytest.approx(want)


def test_exact_simulation_moments() -> None:
    times = np.array([0.25, 1.0])
    s = M2.simulate_state(times, 200_000, np.random.default_rng(0))
    assert s.shape == (2, 200_000, 2)
    chi, xi = s[1, :, 0], s[1, :, 1]
    var_chi = 0.36 * (1 - np.exp(-3.0)) / 3.0
    assert chi.var() == pytest.approx(var_chi, rel=0.02)
    assert xi.var() == pytest.approx(0.0625, rel=0.02)
    cov = 2 * 0 + -0.3 * 0.6 * 0.25 * (1 - np.exp(-1.5)) / 1.5
    assert np.cov(chi, xi)[0, 1] == pytest.approx(cov, abs=0.003)


def test_spot_and_forward_are_martingales() -> None:
    times = np.array([0.3, 0.9])
    fwd = np.array([60.0, 55.0])
    state = M2.simulate_state(times, 200_000, np.random.default_rng(1))
    s = M2.spot(state, times, fwd)
    se = s.std(axis=1) / np.sqrt(s.shape[1])
    assert np.all(np.abs(s.mean(axis=1) - fwd) < 3 * se)
    f = M2.forward_curve(state[0], 0.3, np.array([0.9, 2.0]), np.array([55.0, 40.0]))
    se_f = f.std(axis=0) / np.sqrt(f.shape[0])
    assert np.all(np.abs(f.mean(axis=0) - [55.0, 40.0]) < 3 * se_f)


def test_forward_at_delivery_equals_spot() -> None:
    state = M2.simulate_state(np.array([0.5]), 5, np.random.default_rng(2))[0]
    f = M2.forward_curve(state, 0.5, np.array([0.5]), np.array([50.0]))[:, 0]
    s = M2.spot(state[None], np.array([0.5]), np.array([50.0]))[0]
    np.testing.assert_allclose(f, s)


def test_basis_has_all_monomials_up_to_degree() -> None:
    state = M2.simulate_state(np.array([0.5]), 50, np.random.default_rng(3))[0]
    phi = M2.basis(state, 0.5, "polynomial", 3)
    assert phi.shape == (50, 10)  # 1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3
    np.testing.assert_allclose(phi[:, 0], 1.0)


def test_without_long_term_factor_it_reproduces_the_one_factor_model() -> None:
    m1 = OneFactorModel(kappa=4.0, sigma=0.7)
    m2 = TwoFactorModel(kappa=4.0, sigma_chi=0.7, sigma_xi=0.0, rho=0.0)
    times = np.array([0.2, 0.5, 1.0])
    x = m1.simulate_x(times, 100, np.random.default_rng(4))
    s = m2.simulate_state(times, 100, np.random.default_rng(4))
    np.testing.assert_array_equal(s[..., 0], x)  # bit-identical paths
    np.testing.assert_allclose(s[..., 1], 0.0)

    cfg = load_config("configs/storage_synthetic.yaml")
    inp = build_inputs(cfg, "weekly")
    kw = {"n_levels": 26}
    v1 = lsmc_value(inp, m1, 400, 400, np.random.default_rng(5), np.random.default_rng(6), **kw)
    v2 = lsmc_value(inp, m2, 400, 400, np.random.default_rng(5), np.random.default_rng(6), **kw)
    assert v2.value == pytest.approx(v1.value, rel=1e-9)


def test_two_factor_lsmc_and_rolling_intrinsic_run_and_bound_intrinsic() -> None:
    cfg = load_config("configs/storage_synthetic.yaml")
    inp = build_inputs(cfg, "weekly")
    intr = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, n_levels=26).value
    res = lsmc_value(
        inp, M2, 800, 800, np.random.default_rng(7), np.random.default_rng(8), n_levels=26
    )
    assert res.value >= intr - 3 * res.se
    ri = rolling_intrinsic(inp, M2, 200, np.random.default_rng(9), n_levels=26)
    assert ri.value >= intr - 3 * ri.se
