"""Tests for the seasonal component, the one-factor OU model and its calibration."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gasstorage.config import ModelConfig
from gasstorage.models.calibration import (
    calibrate_one_factor,
    calibrate_windows,
    fit_ou_mle,
    fit_ou_ols,
)
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.models.seasonality import fit_seasonality
from gasstorage.timegrid import year_fraction

# --- time grid -----------------------------------------------------------------------------


def test_year_fraction_is_act_365() -> None:
    dates = pd.DatetimeIndex(["2027-04-01", "2028-03-31"])
    got = year_fraction(dt.date(2026, 11, 17), dates)
    np.testing.assert_allclose(got, [135 / 365, 500 / 365])


# --- seasonality ---------------------------------------------------------------------------


def test_seasonal_fit_recovers_known_fourier_coefficients() -> None:
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
    t = (dates - dates[0]).days.to_numpy() / 365.25
    true = 3.0 + 0.1 * t + 0.2 * np.cos(2 * np.pi * t) - 0.05 * np.sin(2 * np.pi * t)
    true += 0.03 * np.cos(4 * np.pi * t)
    fit = fit_seasonality(dates, true, n_harmonics=2, trend=True)
    np.testing.assert_allclose(fit.coef, [3.0, 0.1, 0.2, -0.05, 0.03, 0.0], atol=1e-10)
    np.testing.assert_allclose(fit.evaluate(dates), true, atol=1e-10)


def test_seasonal_fit_without_trend_has_no_slope_term() -> None:
    dates = pd.date_range("2018-01-01", "2019-12-31", freq="D")
    fit = fit_seasonality(dates, np.full(len(dates), 2.5), n_harmonics=1, trend=False)
    np.testing.assert_allclose(fit.coef, [2.5, 0.0, 0.0], atol=1e-12)


# --- one-factor model ----------------------------------------------------------------------

KAPPA, SIGMA = 3.0, 0.8


def _model() -> OneFactorModel:
    return OneFactorModel(kappa=KAPPA, sigma=SIGMA)


def test_variance_formula() -> None:
    m = _model()
    # V(t, T) = sigma^2 / (2 kappa) (1 - exp(-2 kappa (T - t)))
    want = 0.64 / 6.0 * (1 - np.exp(-6.0 * 0.25))
    assert m.variance(0.5, 0.75) == pytest.approx(want)
    assert m.variance(0.25) == pytest.approx(want)


def test_exact_simulation_has_ou_moments() -> None:
    m = _model()
    times = np.array([0.1, 0.5, 1.0])
    x = m.simulate_x(times, n_paths=200_000, rng=np.random.default_rng(1))
    assert x.shape == (3, 200_000)
    var_true = SIGMA**2 / (2 * KAPPA) * (1 - np.exp(-2 * KAPPA * times))
    se_mean = np.sqrt(var_true / x.shape[1])
    assert np.all(np.abs(x.mean(axis=1)) < 3 * se_mean)
    # SE of a sample variance of Gaussian data: var * sqrt(2 / n)
    assert np.all(np.abs(x.var(axis=1) - var_true) < 3 * var_true * np.sqrt(2 / x.shape[1]))
    # lag correlation of OU: corr(X_s, X_t) = exp(-kappa (t - s)) sqrt(V(0,s) / V(0,t))
    corr = np.corrcoef(x[1], x[2])[0, 1]
    want = np.exp(-KAPPA * 0.5) * np.sqrt(var_true[1] / var_true[2])
    assert corr == pytest.approx(want, abs=0.01)


def test_exact_simulation_is_independent_of_step_size() -> None:
    # Exactness: the distribution of X_1 does not depend on intermediate steps.
    m = _model()
    coarse = m.simulate_x(np.array([1.0]), 100_000, np.random.default_rng(2))[-1]
    fine = m.simulate_x(np.linspace(0.01, 1.0, 100), 100_000, np.random.default_rng(3))[-1]
    se = np.sqrt(coarse.var() * 2 / 100_000)
    assert abs(coarse.var() - fine.var()) < 3 * np.sqrt(2) * se


def test_spot_is_martingale_in_expectation() -> None:
    m = _model()
    times = np.array([0.1, 0.4, 0.9])
    fwd = np.array([60.0, 70.0, 55.0])
    s = m.spot(m.simulate_x(times, 200_000, np.random.default_rng(4)), times, fwd)
    se = s.std(axis=1) / np.sqrt(s.shape[1])
    assert np.all(np.abs(s.mean(axis=1) - fwd) < 3 * se)


def test_implied_forward_is_martingale() -> None:
    m = _model()
    t, T, F0T = 0.3, 1.2, 58.0
    x_t = m.simulate_x(np.array([t]), 200_000, np.random.default_rng(5))[0]
    f = m.forward(x_t, t, T, F0T)
    se = f.std() / np.sqrt(f.size)
    assert abs(f.mean() - F0T) < 3 * se


def test_forward_at_maturity_equals_spot() -> None:
    m = _model()
    x = np.array([-0.4, 0.0, 0.7])
    np.testing.assert_allclose(m.forward(x, 0.6, 0.6, 50.0), m.spot(x[None, :], [0.6], [50.0])[0])


def test_zero_volatility_reproduces_forward_curve() -> None:
    m = OneFactorModel(kappa=2.0, sigma=0.0)
    times = np.array([0.2, 0.7])
    fwd = np.array([40.0, 45.0])
    s = m.spot(m.simulate_x(times, 10, np.random.default_rng(6)), times, fwd)
    np.testing.assert_allclose(s, np.repeat(fwd[:, None], 10, axis=1))


def test_model_from_config() -> None:
    m = OneFactorModel.from_config(ModelConfig(name="one_factor", kappa=4.0, sigma=0.5))
    assert (m.kappa, m.sigma) == (4.0, 0.5)


# --- calibration ---------------------------------------------------------------------------


def _simulate_discrete_ou(
    kappa: float, sigma: float, mu: float, n: int, dt_: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.exp(-kappa * dt_)
    sd = sigma * np.sqrt((1 - a**2) / (2 * kappa))
    y = np.empty(n)
    y[0] = mu
    for i in range(1, n):  # test helper only: one path, sequential by construction
        y[i] = mu + a * (y[i - 1] - mu) + sd * rng.standard_normal()
    return y


@pytest.mark.parametrize("fit", [fit_ou_ols, fit_ou_mle])
def test_calibration_recovers_known_parameters(fit) -> None:
    kappa, sigma, mu, dt_ = 3.0, 0.8, 0.2, 1 / 252
    y = _simulate_discrete_ou(kappa, sigma, mu, n=252 * 20, dt_=dt_, seed=7)
    est = fit(y, dt_)
    assert abs(est.kappa - kappa) < 3 * est.se_kappa
    assert abs(est.sigma - sigma) < 3 * est.se_sigma
    assert abs(est.mu - mu) < 3 * est.se_mu
    assert est.se_kappa > 0 and est.se_sigma > 0
    assert est.half_life_years == pytest.approx(np.log(2) / est.kappa)


def test_ols_and_mle_agree_on_regular_grid() -> None:
    y = _simulate_discrete_ou(3.0, 0.8, 0.2, n=252 * 10, dt_=1 / 252, seed=8)
    ols, mle = fit_ou_ols(y, 1 / 252), fit_ou_mle(y, 1 / 252)
    assert mle.kappa == pytest.approx(ols.kappa, rel=1e-3)
    assert mle.sigma == pytest.approx(ols.sigma, rel=1e-3)


def _synthetic_price_series(seed: int) -> pd.Series:
    dates = pd.bdate_range("2015-01-01", periods=252 * 12)
    t = (dates - dates[0]).days.to_numpy() / 365.25
    y = _simulate_discrete_ou(4.0, 0.6, 0.0, n=len(dates), dt_=1 / 252, seed=seed)
    log_p = np.log(30.0) + 0.15 * np.cos(2 * np.pi * t) + y
    return pd.Series(np.exp(log_p), index=dates, name="price_eur_mwh")


def test_full_pipeline_recovers_parameters_through_seasonality() -> None:
    res = calibrate_one_factor(_synthetic_price_series(9), n_harmonics=2, trend=True)
    est = res.estimate
    assert abs(est.kappa - 4.0) < 3 * est.se_kappa
    assert abs(est.sigma - 0.6) < 3 * est.se_sigma
    assert len(res.residuals) == 252 * 12


def test_calibrate_windows_returns_one_row_per_window() -> None:
    prices = _synthetic_price_series(10)
    windows = {"early": ("2015-01-01", "2019-12-31"), "all": (None, None)}
    table = calibrate_windows(prices, windows)
    assert list(table.index) == ["early", "all"]
    for col in ["kappa", "se_kappa", "sigma", "se_sigma", "half_life_days", "n_obs"]:
        assert col in table.columns
    assert table.loc["all", "n_obs"] == 252 * 12


def test_window_without_mean_reversion_is_reported_not_raised() -> None:
    # explosive log-price y_{i+1} = 1.005 y_i: with a constant-only seasonal fit the AR(1)
    # regression recovers phi = 1.005 exactly, so no OU parameters exist
    dates = pd.bdate_range("2015-01-01", periods=400)
    prices = pd.Series(np.exp(0.01 * 1.005 ** np.arange(400)), index=dates)
    table = calibrate_windows(prices, {"boom": (None, None)}, trend=False, n_harmonics=0)
    assert np.isnan(table.loc["boom", "kappa"])
    assert "no mean reversion" in table.loc["boom", "note"]
    assert table.loc["boom", "n_obs"] == 400


def test_dickey_fuller_statistic_is_reported() -> None:
    table = calibrate_windows(_synthetic_price_series(12), {"all": (None, None)})
    # strongly mean-reverting synthetic data: (phi - 1) / se(phi) far below zero
    assert table.loc["all", "df_tstat"] < -5
    assert table.loc["all", "note"] == ""
