r"""Calibration of the Ornstein-Uhlenbeck component under the historical measure.

The deseasonalised log-price $Y$ (see :mod:`gasstorage.models.seasonality`) is modelled as
$dY_t = \kappa(\mu - Y_t)\,dt + \sigma\,dW_t$, observed on a regular grid of step $\Delta$
(trading days, $\Delta = 1/252$ year). Its exact discretisation is an AR(1):

.. math::
    Y_{i+1} = \alpha + \phi Y_i + \varepsilon_i, \qquad
    \phi = e^{-\kappa\Delta},\quad \alpha = \mu(1-\phi),\quad
    \varepsilon_i \sim \mathcal N\!\Big(0,\ s^2 = \sigma^2\frac{1-e^{-2\kappa\Delta}}{2\kappa}\Big).

**OLS.** Regress $Y_{i+1}$ on $(1, Y_i)$, then invert the mapping:

.. math::
    \hat\kappa = -\frac{\ln\hat\phi}{\Delta},\qquad
    \hat\mu = \frac{\hat\alpha}{1-\hat\phi},\qquad
    \hat\sigma = \hat s\sqrt{\frac{2\hat\kappa}{1-\hat\phi^2}}.

Standard errors by the delta method, from the OLS covariance of $(\hat\alpha, \hat\phi)$ and
$\operatorname{Var}(\hat s) \approx \hat s^2 / (2(n-2))$.

**Exact MLE.** Maximise the Gaussian transition log-likelihood

.. math::
    \ell(\kappa,\mu,\sigma) = -\sum_{i}\Big[\tfrac12\ln(2\pi v) +
        \frac{(Y_{i+1} - \mu - (Y_i-\mu)e^{-\kappa\Delta})^2}{2v}\Big],\qquad
    v = \sigma^2\frac{1-e^{-2\kappa\Delta}}{2\kappa},

starting from the OLS estimate. Standard errors from the inverse observed information (numerical
Hessian of $-\ell$ at the optimum). On a regular grid, MLE and OLS coincide up to the $n$ vs
$n-2$ variance denominator; the MLE is kept because it extends to irregular grids and models
without a closed-form regression.

Half-life of a deviation: $t_{1/2} = \ln 2 / \kappa$.

**Unit-root diagnostic.** The Dickey-Fuller statistic
$(\hat\phi - 1)/\operatorname{se}(\hat\phi)$ is
reported with each estimate. Under the unit-root null it does not follow a Student law; asymptotic
critical values (MacKinnon, 2010) are about $-2.86$ (5 %) with a constant and $-3.41$ (5 %) with a
constant and a linear trend. Daily mean reversion of commodity prices is often weak, so these
statistics tell how much the estimated $\kappa$ can be trusted. If $\hat\phi \ge 1$ no mean
reversion is detected and no OU parameters are returned (:class:`NoMeanReversionError`).

**Measure.** These are historical ($\mathbb P$) estimates. With a constant market price of risk,
Girsanov leaves $\kappa$ and $\sigma$ unchanged and only shifts the mean level, which the pricing
model absorbs by fitting $F(0,t)$ exactly. Remaining gaps (time-varying risk premia, implied
volatility from options, front-month futures used as a spot proxy) are discussed in notebook 02.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gasstorage.models.seasonality import SeasonalFit, fit_seasonality

TRADING_DAYS = 252
DT_TRADING = 1.0 / TRADING_DAYS


@dataclass(frozen=True)
class OUEstimate:
    """Point estimates and standard errors of the OU parameters (annualised)."""

    kappa: float
    mu: float
    sigma: float
    se_kappa: float
    se_mu: float
    se_sigma: float
    n_obs: int
    dt: float
    method: str
    phi: float
    se_phi: float

    @property
    def df_tstat(self) -> float:
        """Dickey-Fuller statistic (phi - 1) / se(phi) of the AR(1) regression."""
        return float((self.phi - 1.0) / self.se_phi)

    @property
    def half_life_years(self) -> float:
        return float(np.log(2.0) / self.kappa)

    @property
    def half_life_days(self) -> float:
        """Half-life in calendar days (365 per year)."""
        return 365.0 * self.half_life_years


def _jacobian(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, rel: float = 1e-6
) -> np.ndarray:
    h = rel * np.maximum(np.abs(x), 1e-4)
    cols = []
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = h[j]
        cols.append((f(x + e) - f(x - e)) / (2 * h[j]))
    return np.column_stack(cols)


def _hessian(f: Callable[[np.ndarray], float], x: np.ndarray, rel: float = 1e-4) -> np.ndarray:
    h = rel * np.maximum(np.abs(x), 1e-2)
    n = len(x)
    hess = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            val = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4 * h[i] * h[j]
            )
            hess[i, j] = hess[j, i] = val
    return hess


class NoMeanReversionError(ValueError):
    """The AR(1) coefficient is not in (0, 1): the OU model cannot be fitted."""

    def __init__(self, phi: float, se_phi: float) -> None:
        super().__init__(f"AR(1) coefficient {phi:.4f} outside (0, 1): no mean reversion detected")
        self.phi = phi
        self.se_phi = se_phi


def _ar1_to_ou(theta: np.ndarray, dt: float) -> np.ndarray:
    alpha, phi, s = theta
    kappa = -np.log(phi) / dt
    mu = alpha / (1.0 - phi)
    sigma = s * np.sqrt(2.0 * kappa / (1.0 - phi**2))
    return np.array([kappa, mu, sigma])


def fit_ou_ols(y: np.ndarray, dt: float = DT_TRADING) -> OUEstimate:
    """AR(1) OLS estimate of the OU parameters with delta-method standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    target = y[1:]
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    resid = target - x @ beta
    n = len(target)
    s2 = resid @ resid / (n - 2)
    alpha, phi = beta
    cov = np.zeros((3, 3))
    cov[:2, :2] = s2 * np.linalg.inv(x.T @ x)
    se_phi = float(np.sqrt(cov[1, 1]))
    if not 0.0 < phi < 1.0:
        raise NoMeanReversionError(float(phi), se_phi)
    cov[2, 2] = s2 / (2.0 * (n - 2))
    theta = np.array([alpha, phi, np.sqrt(s2)])
    kappa, mu, sigma = _ar1_to_ou(theta, dt)
    jac = _jacobian(lambda th: _ar1_to_ou(th, dt), theta)
    se = np.sqrt(np.diag(jac @ cov @ jac.T))
    return OUEstimate(
        kappa, mu, sigma, *se, n_obs=len(y), dt=dt, method="ols", phi=float(phi), se_phi=se_phi
    )


def _neg_loglik(params: np.ndarray, y: np.ndarray, dt: float) -> float:
    kappa, mu, sigma = params
    if kappa <= 0 or sigma <= 0:
        return np.inf
    a = np.exp(-kappa * dt)
    v = sigma**2 * -np.expm1(-2.0 * kappa * dt) / (2.0 * kappa)
    r = y[1:] - mu - (y[:-1] - mu) * a
    return float(0.5 * len(r) * np.log(2.0 * np.pi * v) + (r @ r) / (2.0 * v))


def fit_ou_mle(y: np.ndarray, dt: float = DT_TRADING) -> OUEstimate:
    """Exact Gaussian MLE of the OU parameters; standard errors from the observed information."""
    y = np.asarray(y, dtype=float)
    start = fit_ou_ols(y, dt)

    def nll_log(p: np.ndarray) -> float:
        return _neg_loglik(np.array([np.exp(p[0]), p[1], np.exp(p[2])]), y, dt)

    p0 = np.array([np.log(start.kappa), start.mu, np.log(start.sigma)])
    res = minimize(
        nll_log,
        p0,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000},
    )
    if not res.success:
        raise RuntimeError(f"MLE did not converge: {res.message}")
    theta = np.array([np.exp(res.x[0]), res.x[1], np.exp(res.x[2])])
    hess = _hessian(lambda th: _neg_loglik(th, y, dt), theta)
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    return OUEstimate(
        *theta, *se, n_obs=len(y), dt=dt, method="mle", phi=start.phi, se_phi=start.se_phi
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Seasonal fit + OU estimate on one calibration window."""

    estimate: OUEstimate
    seasonal: SeasonalFit
    residuals: pd.Series


def calibrate_one_factor(
    prices: pd.Series,
    start: str | None = None,
    end: str | None = None,
    n_harmonics: int = 2,
    trend: bool = True,
    method: str = "mle",
) -> CalibrationResult:
    """Deseasonalise $\\ln S$ on ``[start, end]`` and fit the OU component.

    ``method`` is ``"mle"`` (default) or ``"ols"``.
    """
    window = prices.loc[start:end]
    log_p = np.log(window.to_numpy(dtype=float))
    seasonal = fit_seasonality(window.index, log_p, n_harmonics=n_harmonics, trend=trend)
    resid = log_p - seasonal.evaluate(window.index)
    fit = {"mle": fit_ou_mle, "ols": fit_ou_ols}[method]
    estimate = fit(resid, DT_TRADING)
    return CalibrationResult(estimate, seasonal, pd.Series(resid, index=window.index, name="y"))


def calibrate_windows(
    prices: pd.Series,
    windows: Mapping[str, tuple[str | None, str | None]],
    **kwargs: object,
) -> pd.DataFrame:
    """Calibrate on several windows; one row per window with estimates and standard errors."""
    rows = {}
    for name, (start, end) in windows.items():
        window = prices.loc[start:end]
        row: dict[str, object] = {
            "start": window.index[0].date(),
            "end": window.index[-1].date(),
            "n_obs": len(window),
        }
        try:
            res = calibrate_one_factor(prices, start, end, **kwargs)  # type: ignore[arg-type]
        except NoMeanReversionError as err:
            nan = float("nan")
            row |= dict.fromkeys(
                ["kappa", "se_kappa", "sigma", "se_sigma", "mu", "se_mu", "half_life_days"], nan
            )
            row |= {
                "df_tstat": (err.phi - 1.0) / err.se_phi,
                "seasonal_amplitude": nan,
                "note": f"no mean reversion (phi = {err.phi:.4f})",
            }
        else:
            e = res.estimate
            row |= {
                "kappa": e.kappa,
                "se_kappa": e.se_kappa,
                "sigma": e.sigma,
                "se_sigma": e.se_sigma,
                "mu": e.mu,
                "se_mu": e.se_mu,
                "half_life_days": e.half_life_days,
                "df_tstat": e.df_tstat,
                "seasonal_amplitude": res.seasonal.amplitude(),
                "note": "",
            }
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")
