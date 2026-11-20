r"""Principal components of daily forward curve moves, compared with the price models.

**Data.** For fixed delivery months $T_1, \dots, T_n$ (constant contracts, not constant maturity),
the daily log-returns $r_{t,i} = \ln F(t, T_i) - \ln F(t-1, T_i)$ over $N$ sessions give the sample
covariance $\hat\Sigma$. Its eigen-decomposition $\hat\Sigma = \sum_k \lambda_k v_k v_k^\top$ gives
the share of variance of component $k$, $\lambda_k / \sum_j \lambda_j$, and its loadings $v_k$
(sign convention: PC1 loadings sum to a positive number, the other components are positive on the
longest maturity). Confidence intervals on the shares come from an i.i.d. bootstrap over days,
which ignores the (weak) autocorrelation of daily returns.

**Models.** With $\tau_i = T_i - t$, the instantaneous covariance of $d\ln F(t, T_i)$ per year is

.. math::
    \Sigma_{ij} = \sigma_\chi^2 e^{-\kappa(\tau_i + \tau_j)} + \sigma_\xi^2
        + \rho\,\sigma_\chi\sigma_\xi \left(e^{-\kappa\tau_i} + e^{-\kappa\tau_j}\right)

in the Schwartz-Smith two-factor model, and $\sigma^2 e^{-\kappa(\tau_i+\tau_j)}$ (rank one: one
component explains everything) in the one-factor model. $\tau_i$ is measured from the midpoint of
the sample window; over a 45-day window this moves $\tau$ by at most about 0.06 years.

**Spread.** A strip is approximated in log terms by a day-weighted average of its months, so the
log-spread between a long strip $L$ and a short strip $S$ has weights $w_i = d_i / \sum_{L} d$ on
$L$ and $-d_i / \sum_{S} d$ on $S$ ($d_i$ = days in month $i$), with model volatility
$\sqrt{w^\top \Sigma w}$ and sample volatility $\hat\sigma = \operatorname{std}(r w)\sqrt{252}$,
$\operatorname{SE}(\hat\sigma) \approx \hat\sigma / \sqrt{2(N-1)}$ for Gaussian returns.

**Caveat.** Exchanges settle illiquid months from the quarter or season they belong to, so some
consecutive months move in lockstep (`locked_groups`): the curve has fewer independent series than
columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gasstorage.models.vol_term_structure import TRADING_DAYS


def curve_returns(panel: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """Daily log-returns of the delivery months in ``[first, last]``, sessions with all prices."""
    cols = panel.columns[
        (panel.columns >= pd.Period(first, "M")) & (panel.columns <= pd.Period(last, "M"))
    ]
    return np.log(panel[cols]).diff().dropna(how="any")


def locked_groups(returns: pd.DataFrame, tol: float = 1e-3) -> list[list[pd.Period]]:
    """Runs of consecutive months whose daily returns never differ by more than ``tol``."""
    groups: list[list[pd.Period]] = [[returns.columns[0]]]
    for prev, cur in zip(returns.columns[:-1], returns.columns[1:], strict=True):
        if float((returns[cur] - returns[prev]).abs().max()) < tol:
            groups[-1].append(cur)
        else:
            groups.append([cur])
    return groups


@dataclass(frozen=True)
class PCAResult:
    eigenvalues: np.ndarray  # annualised, descending
    shares: np.ndarray  # share of total variance, descending
    loadings: pd.DataFrame  # index = delivery month, columns PC1..PCk
    share_ci: pd.DataFrame  # index PC1..PCk, columns lo / hi (bootstrap percentiles)
    n_obs: int


def _eig(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = np.maximum(w[order], 0.0), v[:, order]
    signs = np.sign(v[-1, :])
    signs[0] = np.sign(v[:, 0].sum())
    signs[signs == 0] = 1.0
    return w, v * signs


def explained_shares(cov: np.ndarray) -> np.ndarray:
    """Share of total variance of each principal component of ``cov``, descending."""
    w = _eig(cov)[0]
    return w / w.sum()


def curve_pca(
    returns: pd.DataFrame,
    n_components: int = 3,
    n_boot: int = 2000,
    seed: int = 2468,
    level: float = 0.95,
) -> PCAResult:
    """PCA of the sample covariance of daily log-returns, with bootstrap intervals on the shares."""
    x = returns.to_numpy()
    w, v = _eig(np.cov(x, rowvar=False) * TRADING_DAYS)
    shares = w / w.sum()
    names = [f"PC{k + 1}" for k in range(n_components)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = np.array([explained_shares(np.cov(x[i], rowvar=False))[:n_components] for i in idx])
    a = (1.0 - level) / 2
    ci = pd.DataFrame(
        {"lo": np.quantile(boot, a, axis=0), "hi": np.quantile(boot, 1 - a, axis=0)}, index=names
    )
    loadings = pd.DataFrame(v[:, :n_components], index=returns.columns, columns=names)
    return PCAResult(w, shares, loadings, ci, len(x))


def delivery_tau(months: pd.PeriodIndex, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Years from the midpoint of the sample window to the start of each delivery month."""
    mid = sessions.min() + (sessions.max() - sessions.min()) / 2
    starts = months.to_timestamp(how="start")
    return np.asarray((starts - mid.normalize()).days, dtype=float) / 365.0


def model_covariance(
    tau: np.ndarray, kappa: float, sigma_chi: float, sigma_xi: float = 0.0, rho: float = 0.0
) -> np.ndarray:
    """Annualised instantaneous covariance of forward log-returns (1F if ``sigma_xi = 0``)."""
    e = np.exp(-kappa * np.asarray(tau, dtype=float))
    return (
        sigma_chi**2 * np.outer(e, e)
        + sigma_xi**2
        + rho * sigma_chi * sigma_xi * (e[:, None] + e[None, :])
    )


def spread_weights(
    months: pd.PeriodIndex, long: tuple[str, str], short: tuple[str, str]
) -> np.ndarray:
    """Day-weighted log-spread weights: $+1$ in total on ``long``, $-1$ on ``short``."""
    months = pd.PeriodIndex(months, freq="M")
    days = months.days_in_month.to_numpy().astype(float)

    def mask(span: tuple[str, str]) -> np.ndarray:
        return np.asarray((months >= pd.Period(span[0], "M")) & (months <= pd.Period(span[1], "M")))

    ml, ms = mask(long), mask(short)
    return np.where(ml, days / days[ml].sum(), 0.0) - np.where(ms, days / days[ms].sum(), 0.0)


def spread_vol(returns: pd.DataFrame, weights: np.ndarray) -> tuple[float, float]:
    """Annualised sample volatility of the weighted log-return and its standard error."""
    s = returns.to_numpy() @ weights
    vol = float(np.std(s, ddof=1) * np.sqrt(TRADING_DAYS))
    return vol, vol / np.sqrt(2 * (len(s) - 1))
