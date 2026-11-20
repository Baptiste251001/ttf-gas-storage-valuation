"""Tests for the principal component analysis of forward curve moves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gasstorage.models.curve_pca import (
    curve_pca,
    curve_returns,
    delivery_tau,
    locked_groups,
    model_covariance,
    spread_vol,
    spread_weights,
)
from gasstorage.models.vol_term_structure import TRADING_DAYS, two_factor_forward_vol

MONTHS = pd.period_range("2027-01", "2027-06", freq="M")


def _panel(returns: np.ndarray) -> pd.DataFrame:
    days = pd.bdate_range("2026-10-01", periods=len(returns) + 1)
    prices = 50.0 * np.exp(np.vstack([np.zeros(returns.shape[1]), np.cumsum(returns, axis=0)]))
    return pd.DataFrame(prices, index=days, columns=MONTHS)


def test_curve_returns_selects_months_and_takes_log_differences() -> None:
    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_normal((20, len(MONTHS)))
    out = curve_returns(_panel(r), "2027-02", "2027-05")
    assert list(out.columns) == list(MONTHS[1:5])
    np.testing.assert_allclose(out.to_numpy(), r[:, 1:5], atol=1e-12)


def test_locked_groups_finds_months_that_move_together() -> None:
    rng = np.random.default_rng(1)
    base = 0.01 * rng.standard_normal((30, 3))
    r = np.column_stack([base[:, 0], base[:, 0], base[:, 1], base[:, 2], base[:, 2], base[:, 2]])
    groups = locked_groups(curve_returns(_panel(r), "2027-01", "2027-06"))
    assert [len(g) for g in groups] == [2, 1, 3]
    assert groups[0] == [MONTHS[0], MONTHS[1]]


def test_pca_recovers_a_two_factor_covariance() -> None:
    # level factor (variance 4) and slope factor (variance 1), orthogonal loadings
    n = len(MONTHS)
    level = np.ones(n) / np.sqrt(n)
    slope = np.linspace(-1, 1, n)
    slope /= np.linalg.norm(slope)
    rng = np.random.default_rng(2)
    f = rng.standard_normal((20_000, 2)) * np.array([2.0, 1.0]) * 0.01
    r = f[:, :1] * level + f[:, 1:] * slope
    res = curve_pca(curve_returns(_panel(r), "2027-01", "2027-06"), n_boot=200, seed=3)
    np.testing.assert_allclose(res.shares[:2], [0.8, 0.2], atol=0.01)
    assert res.shares[2] < 1e-10
    # sign convention: PC1 loadings positive, PC2 positive on the last maturity
    np.testing.assert_allclose(res.loadings["PC1"], level, atol=0.01)
    np.testing.assert_allclose(res.loadings["PC2"], slope, atol=0.01)
    lo, hi = res.share_ci.loc["PC1", "lo"], res.share_ci.loc["PC1", "hi"]
    assert lo < res.shares[0] < hi


def test_pca_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(4)
    ret = curve_returns(_panel(0.01 * rng.standard_normal((40, 6))), "2027-01", "2027-06")
    a, b = curve_pca(ret, n_boot=100, seed=7), curve_pca(ret, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(a.share_ci, b.share_ci)


def test_model_covariance_one_factor_is_rank_one() -> None:
    tau = np.array([0.2, 0.5, 1.0, 1.5])
    c = model_covariance(tau, kappa=0.4, sigma_chi=0.7)
    e = np.exp(-0.4 * tau)
    np.testing.assert_allclose(c, 0.49 * np.outer(e, e))
    assert np.linalg.matrix_rank(c) == 1


def test_model_covariance_two_factor_diagonal_matches_forward_vol() -> None:
    tau = np.array([0.1, 0.6, 1.2, 2.0])
    c = model_covariance(tau, kappa=0.46, sigma_chi=0.74, sigma_xi=0.27, rho=-0.31)
    np.testing.assert_allclose(
        np.sqrt(np.diag(c)), two_factor_forward_vol(tau, 0.46, 0.74, 0.27, -0.31)
    )
    assert np.linalg.matrix_rank(c) == 2
    np.testing.assert_allclose(c, c.T)


def test_delivery_tau_from_window_midpoint() -> None:
    days = pd.DatetimeIndex(["2026-10-01", "2026-10-31"])
    tau = delivery_tau(pd.PeriodIndex(["2027-01", "2027-04"], freq="M"), days)
    mid = pd.Timestamp("2026-10-16")
    want = [
        (pd.Timestamp("2027-01-01") - mid).days / 365,
        (pd.Timestamp("2027-04-01") - mid).days / 365,
    ]
    np.testing.assert_allclose(tau, want)


def test_spread_vol_data_and_model_agree_on_simulated_returns() -> None:
    tau = delivery_tau(MONTHS, pd.DatetimeIndex(["2026-10-01", "2026-10-31"]))
    cov = model_covariance(tau, kappa=0.5, sigma_chi=0.7, sigma_xi=0.3, rho=-0.3)
    rng = np.random.default_rng(5)
    r = rng.multivariate_normal(np.zeros(len(MONTHS)), cov / TRADING_DAYS, size=20_000)
    ret = curve_returns(_panel(r), "2027-01", "2027-06")
    w = spread_weights(ret.columns, long=("2027-04", "2027-06"), short=("2027-01", "2027-03"))
    np.testing.assert_allclose(w.sum(), 0.0, atol=1e-12)
    vol, se = spread_vol(ret, w)
    want = float(np.sqrt(w @ cov @ w))
    assert vol == pytest.approx(want, abs=3 * se)
    assert se == pytest.approx(vol / np.sqrt(2 * (len(ret) - 1)))
