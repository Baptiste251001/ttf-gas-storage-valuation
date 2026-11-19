"""Tests for TTF contract tickers and the futures volatility term structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gasstorage.data.ttf import contract_ticker
from gasstorage.models.vol_term_structure import (
    fit_one_factor_kappa,
    fit_two_factor,
    two_factor_forward_vol,
    vol_ratio_table,
)


def test_contract_ticker_follows_yahoo_nymex_convention() -> None:
    # Convention checked against live Yahoo data: TTFF27.NYM settled at 85.30 on 2026-11-17
    # (EEX Jan-27 settlement: 85.35).
    assert contract_ticker(pd.Period("2027-01", "M")) == "TTFF27.NYM"
    assert contract_ticker(pd.Period("2026-12", "M")) == "TTFZ26.NYM"
    assert contract_ticker(pd.Period("2028-03", "M")) == "TTFH28.NYM"


def test_two_factor_forward_vol_formula() -> None:
    # sigma(tau)^2 = s_chi^2 e^{-2 kappa tau} + s_xi^2 + 2 rho s_chi s_xi e^{-kappa tau}
    got = two_factor_forward_vol(
        np.array([0.0, 1.0]), kappa=1.0, sigma_chi=0.6, sigma_xi=0.2, rho=0.5
    )
    want0 = np.sqrt(0.36 + 0.04 + 2 * 0.5 * 0.12)
    e = np.exp(-1.0)
    want1 = np.sqrt(0.36 * e**2 + 0.04 + 2 * 0.5 * 0.12 * e)
    np.testing.assert_allclose(got, [want0, want1])
    one = two_factor_forward_vol(np.array([2.0]), kappa=0.5, sigma_chi=0.7, sigma_xi=0.0, rho=0.0)
    np.testing.assert_allclose(one, 0.7 * np.exp(-1.0))


def _simulated_market(kappa, sigma_chi, sigma_xi, rho, seed=0):
    """Daily log-forwards of three contracts and a constant-maturity front from a 2F model."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", "2026-11-30")
    dt = 1 / 252
    z1 = rng.standard_normal(len(dates))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(len(dates))
    starts = {"A": "2026-12-01", "B": "2027-01-01", "C": "2027-02-01"}
    contracts = {}
    for name, start in starts.items():
        tau = (pd.Timestamp(start) - dates).days.to_numpy() / 365
        r = np.sqrt(dt) * (sigma_chi * np.exp(-kappa * tau) * z1 + sigma_xi * z2)
        contracts[name] = pd.Series(50 * np.exp(np.cumsum(r)), index=dates)
    tau_f = 0.05
    rf = np.sqrt(dt) * (sigma_chi * np.exp(-kappa * tau_f) * z1 + sigma_xi * z2)
    front = pd.Series(50 * np.exp(np.cumsum(rf)), index=dates)
    return pd.DataFrame(contracts), {k: pd.Timestamp(v) for k, v in starts.items()}, front, tau_f


def test_kappa_q_is_recovered_from_simulated_one_factor_market() -> None:
    contracts, starts, front, tau_f = _simulated_market(0.4, 0.7, 0.0, 0.0, seed=1)
    table = vol_ratio_table(contracts, starts, front)
    assert {"tau", "ratio", "se", "n"}.issubset(table.columns)
    fit = fit_one_factor_kappa(table, tau_front=tau_f)
    assert abs(fit.kappa - 0.4) < 3 * fit.se_kappa
    assert fit.se_kappa < 0.2


def test_two_factor_ratio_fit_recovers_parameters() -> None:
    contracts, starts, front, tau_f = _simulated_market(2.0, 0.6, 0.25, 0.3, seed=2)
    table = vol_ratio_table(contracts, starts, front)
    fit = fit_two_factor(
        table,
        front_vol=float(two_factor_forward_vol(np.array([tau_f]), 2.0, 0.6, 0.25, 0.3)[0]),
        tau_front=tau_f,
    )
    assert fit.kappa == pytest.approx(2.0, rel=0.35)
    assert fit.sigma_xi == pytest.approx(0.25, rel=0.25)
    assert fit.sigma_chi == pytest.approx(0.6, rel=0.25)
    assert fit.rho == pytest.approx(0.3, abs=0.25)


def test_two_factor_correlation_formula_is_one_for_a_single_factor() -> None:
    from gasstorage.models.vol_term_structure import two_factor_correlation

    one = two_factor_correlation(
        np.array([0.5, 3.0]), 0.04, kappa=1.0, sigma_chi=0.7, sigma_xi=0.0, rho=0.0
    )
    np.testing.assert_allclose(one, 1.0)
    # long maturity, independent factors: corr -> sigma_xi / sigma_F(tau_f)
    two = two_factor_correlation(
        np.array([50.0]), 0.0, kappa=1.0, sigma_chi=0.6, sigma_xi=0.8, rho=0.0
    )
    np.testing.assert_allclose(two, 0.8 / 1.0)
