r"""Term structure of futures volatility and the risk-neutral mean reversion $\kappa_{\mathbb Q}$.

In the Schwartz-Smith (2000) two-factor model, fitted to the curve, the instantaneous volatility of
the forward with time to delivery $\tau$ is

.. math::
    \sigma_F(\tau)^2 = \sigma_\chi^2 e^{-2\kappa\tau} + \sigma_\xi^2
        + 2\rho\,\sigma_\chi\sigma_\xi e^{-\kappa\tau},

which reduces to $\sigma e^{-\kappa\tau}$ in the one-factor model ($\sigma_\xi = 0$). Volatility is
invariant under an equivalent change of measure, so realised futures volatilities identify the
**risk-neutral** $\kappa$, unlike the mean reversion of price levels.

**Estimator.** Realised volatility moves a lot over time (2023 versus 2025), and for one contract
the time to delivery and the calendar date move together. To separate maturity from regime, each
contract's volatility is measured relative to the front month **over the same days**:

.. math::
    R(\tau) = \frac{\hat\sigma_{\text{contract}}(\tau)}{\hat\sigma_{\text{front}}}
    \approx \frac{\sigma_F(\tau)}{\sigma_F(\tau_f)},

computed in bins of $\tau$ from daily log-returns. For bivariate Gaussian returns with correlation
$c$, $\operatorname{Var}(\ln\hat\sigma_1 - \ln\hat\sigma_2) \approx (1-c^2)/n$, so
$\operatorname{SE}(R) \approx R\sqrt{(1-c^2)/n}$.

* One factor: $\ln R(\tau) = -\kappa(\tau - \tau_f)$, weighted least squares through the origin.
* Two factors: the ratio curve alone cannot separate two factors (a single smooth decay fits it),
  but the **correlation** with the front month can: it is 1 in any one-factor model and, with two
  factors,

  .. math::
      \operatorname{corr}(\tau,\tau_f) = \frac{\sigma_\chi^2 e^{-\kappa(\tau+\tau_f)} + \sigma_\xi^2
          + \rho\sigma_\chi\sigma_\xi\,(e^{-\kappa\tau} + e^{-\kappa\tau_f})}
          {\sigma_F(\tau)\,\sigma_F(\tau_f)}.

  $(\kappa, \sigma_\xi/\sigma_\chi, \rho)$ are fitted jointly to $R(\tau)$ and to the Fisher
  transform of the correlations ($\operatorname{SE}(\operatorname{atanh}\hat c) = 1/\sqrt{n-3}$);
  the scale $\sigma_\chi$ then matches the front-month volatility $\sigma_F(\tau_f)$.

Standard errors are scaled by the residual dispersion (returns of overlapping contracts are
correlated, so the nominal ones are optimistic).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

TRADING_DAYS = 252
TAU_FRONT = 0.04  # average time to delivery of the front-month contract (years)


def two_factor_forward_vol(
    tau: np.ndarray, kappa: float, sigma_chi: float, sigma_xi: float, rho: float
) -> np.ndarray:
    """Instantaneous volatility $\\sigma_F(\\tau)$ of a forward with time to delivery ``tau``."""
    e = np.exp(-kappa * np.asarray(tau, dtype=float))
    var = sigma_chi**2 * e**2 + sigma_xi**2 + 2 * rho * sigma_chi * sigma_xi * e
    return np.sqrt(np.maximum(var, 0.0))


def two_factor_correlation(
    tau: np.ndarray, tau_front: float, kappa: float, sigma_chi: float, sigma_xi: float, rho: float
) -> np.ndarray:
    """Instantaneous correlation of forwards with times to delivery ``tau`` and ``tau_front``."""
    tau = np.asarray(tau, dtype=float)
    e, ef = np.exp(-kappa * tau), np.exp(-kappa * tau_front)
    cov = sigma_chi**2 * e * ef + sigma_xi**2 + rho * sigma_chi * sigma_xi * (e + ef)
    s1 = two_factor_forward_vol(tau, kappa, sigma_chi, sigma_xi, rho)
    s2 = two_factor_forward_vol(np.array([tau_front]), kappa, sigma_chi, sigma_xi, rho)
    return cov / (s1 * s2)


def vol_ratio_table(
    contracts: pd.DataFrame,
    delivery_starts: Mapping[str, pd.Timestamp],
    front: pd.Series,
    bin_width: float = 1 / 6,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Volatility of each contract relative to the front month, by bin of time to delivery."""
    rf = np.log(front).diff().rename("f")
    pieces = []
    for name in contracts.columns:
        rc = np.log(contracts[name].dropna()).diff().rename("c")
        joint = pd.concat([rc, rf], axis=1, join="inner").dropna()
        joint["tau"] = (pd.Timestamp(delivery_starts[name]) - joint.index).days / 365.0
        pieces.append(joint[joint["tau"] > 0])
    data = pd.concat(pieces)
    edges = np.arange(0.0, data["tau"].max() + bin_width, bin_width)
    rows = []
    for _, g in data.groupby(pd.cut(data["tau"], edges), observed=True):
        if len(g) < min_obs:
            continue
        vc, vf = g["c"].std(ddof=1), g["f"].std(ddof=1)
        corr = float(np.corrcoef(g["c"], g["f"])[0, 1])
        ratio = vc / vf
        rows.append(
            {
                "tau": g["tau"].mean(),
                "ratio": ratio,
                "se": ratio * np.sqrt(max(1.0 - corr**2, 1e-6) / len(g)),
                "n": len(g),
                "corr": corr,
                "vol_contract": vc * np.sqrt(TRADING_DAYS),
                "vol_front": vf * np.sqrt(TRADING_DAYS),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class KappaFit:
    kappa: float
    se_kappa: float
    n_bins: int

    @property
    def half_life_years(self) -> float:
        return float(np.log(2.0) / self.kappa)


def fit_one_factor_kappa(table: pd.DataFrame, tau_front: float = TAU_FRONT) -> KappaFit:
    """Weighted least squares of $\\ln R = -\\kappa(\\tau - \\tau_f)$."""
    y = np.log(table["ratio"].to_numpy())
    x = -(table["tau"].to_numpy() - tau_front)
    w = (table["ratio"].to_numpy() / table["se"].to_numpy()) ** 2  # 1 / Var(ln R)
    kappa = float(np.sum(w * x * y) / np.sum(w * x**2))
    resid = y - kappa * x
    s2 = max(float(np.sum(w * resid**2) / (len(y) - 1)), 1.0)
    se = float(np.sqrt(s2 / np.sum(w * x**2)))
    return KappaFit(kappa=kappa, se_kappa=se, n_bins=len(y))


@dataclass(frozen=True)
class TwoFactorFit:
    kappa: float
    sigma_chi: float
    sigma_xi: float
    rho: float
    se: dict[str, float]
    n_bins: int


def fit_two_factor(
    table: pd.DataFrame, front_vol: float, tau_front: float = TAU_FRONT
) -> TwoFactorFit:
    """Fit $(\\kappa, \\sigma_\\chi, \\sigma_\\xi, \\rho)$ to the ratio and correlation curves."""
    tau = table["tau"].to_numpy()
    ratio = table["ratio"].to_numpy()
    se = table["se"].to_numpy()
    z_obs = np.arctanh(np.clip(table["corr"].to_numpy(), -0.9999, 0.9999))
    z_w = np.sqrt(np.maximum(table["n"].to_numpy() - 3, 1))
    front = np.array([tau_front])

    def residuals(p: np.ndarray) -> np.ndarray:
        kappa, lam, rho = p
        r_model = two_factor_forward_vol(tau, kappa, 1.0, lam, rho) / two_factor_forward_vol(
            front, kappa, 1.0, lam, rho
        )
        c_model = two_factor_correlation(tau, tau_front, kappa, 1.0, lam, rho)
        z_model = np.arctanh(np.clip(c_model, -0.9999, 0.9999))
        return np.concatenate([(r_model - ratio) / se, (z_model - z_obs) * z_w])

    res = least_squares(
        residuals, x0=np.array([1.0, 0.3, 0.0]), bounds=([1e-3, 0.0, -0.99], [50.0, 10.0, 0.99])
    )
    kappa, lam, rho = res.x
    scale = front_vol / float(two_factor_forward_vol(front, kappa, 1.0, lam, rho)[0])
    s2 = max(float(np.sum(res.fun**2) / max(2 * len(tau) - 3, 1)), 1.0)
    try:
        se_p = np.sqrt(np.diag(np.linalg.inv(res.jac.T @ res.jac) * s2))
    except np.linalg.LinAlgError:
        se_p = np.full(3, np.nan)
    return TwoFactorFit(
        kappa=float(kappa),
        sigma_chi=float(scale),
        sigma_xi=float(scale * lam),
        rho=float(rho),
        se={"kappa": float(se_p[0]), "sigma_xi/sigma_chi": float(se_p[1]), "rho": float(se_p[2])},
        n_bins=len(tau),
    )
