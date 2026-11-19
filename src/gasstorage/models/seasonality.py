r"""Deterministic seasonal component of the log-price.

Historical log-prices are decomposed as $\ln S_t = f(t) + Y_t$ with

.. math::
    f(t) = a + b\,t + \sum_{k=1}^{K} \big(c_k \cos(2\pi k t) + d_k \sin(2\pi k t)\big),

$t$ in years (365.25 days) from the first observation. The coefficients are fitted by ordinary
least squares; the residual $Y_t$ is the stochastic component whose mean-reversion is calibrated in
:mod:`gasstorage.models.calibration`.

This decomposition is used for *calibration only*. In the pricing model the deterministic level
and seasonality come from the forward curve $F(0, t)$, which is matched exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_SEASONAL_YEAR = 365.25


def _years(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return (pd.DatetimeIndex(dates) - origin).days.to_numpy(dtype=float) / DAYS_PER_SEASONAL_YEAR


def fourier_design(t: np.ndarray, n_harmonics: int, trend: bool = True) -> np.ndarray:
    """Design matrix with columns ``[1, t (if trend), cos(2 pi k t), sin(2 pi k t), ...]``."""
    cols = [np.ones_like(t)]
    if trend:
        cols.append(t)
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    return np.column_stack(cols)


@dataclass(frozen=True)
class SeasonalFit:
    """Fitted seasonal component. ``coef`` follows the column order of :func:`fourier_design`."""

    coef: np.ndarray
    n_harmonics: int
    trend: bool
    origin: pd.Timestamp

    def evaluate(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Seasonal log-level $f(t)$ at the given dates."""
        x = fourier_design(_years(dates, self.origin), self.n_harmonics, self.trend)
        return x @ self.coef

    def amplitude(self) -> float:
        """Peak-to-trough amplitude of the periodic part over one year (log units)."""
        t = np.linspace(0.0, 1.0, 366)
        start = 2 if self.trend else 1
        periodic = fourier_design(t, self.n_harmonics, trend=False)[:, 1:] @ self.coef[start:]
        return float(periodic.max() - periodic.min())


def fit_seasonality(
    dates: pd.DatetimeIndex, log_prices: np.ndarray, n_harmonics: int = 2, trend: bool = True
) -> SeasonalFit:
    """OLS fit of the trend + Fourier seasonal component to log-prices."""
    dates = pd.DatetimeIndex(dates)
    origin = dates[0]
    x = fourier_design(_years(dates, origin), n_harmonics, trend)
    coef, *_ = np.linalg.lstsq(x, np.asarray(log_prices, dtype=float), rcond=None)
    return SeasonalFit(coef=coef, n_harmonics=n_harmonics, trend=trend, origin=origin)
