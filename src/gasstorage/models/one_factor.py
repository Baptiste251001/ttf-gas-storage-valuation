r"""One-factor mean-reverting model of the TTF spot price, fitted exactly to the forward curve.

.. math::
    \ln S_t = \ln F(0,t) + X_t - \tfrac12 V(0,t), \qquad
    dX_t = -\kappa X_t\,dt + \sigma\,dW_t, \quad X_0 = 0,

with $V(t,T) = \operatorname{Var}(X_T \mid X_t)
= \frac{\sigma^2}{2\kappa}\big(1 - e^{-2\kappa(T-t)}\big)$.
Since $X_t \sim \mathcal N(0, V(0,t))$, $\mathbb E[S_t] = F(0,t)$: the model reproduces the curve.

**Exact simulation.** Conditionally on $X_s$, $X_t$ is Gaussian:

.. math::
    X_t = e^{-\kappa(t-s)} X_s + \sqrt{V(s,t)}\; Z, \qquad Z \sim \mathcal N(0,1),

so paths are simulated on any grid without discretisation error (no Euler scheme).

**Implied forward.** For $t \le T$, using
$X_T \mid X_t \sim \mathcal N(e^{-\kappa(T-t)}X_t,\, V(t,T))$,

.. math::
    F(t,T) = \mathbb E_t[S_T]
           = F(0,T)\exp\!\Big(e^{-\kappa(T-t)}X_t + \tfrac12 V(t,T) - \tfrac12 V(0,T)\Big).

It is a martingale in $t$ because $e^{-2\kappa(T-t)}V(0,t) + V(t,T) = V(0,T)$.
Time is in years (ACT/365, see :mod:`gasstorage.timegrid`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gasstorage.config import ModelConfig


@dataclass(frozen=True)
class OneFactorModel:
    """OU log-price deviation from the forward curve.

    ``kappa`` in 1/year, ``sigma`` in 1/sqrt(year).
    """

    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> OneFactorModel:
        return cls(kappa=cfg.kappa, sigma=cfg.sigma)

    def variance(self, t: ArrayLike, T: ArrayLike | None = None) -> np.ndarray:
        r"""$V(t,T) = \frac{\sigma^2}{2\kappa}(1 - e^{-2\kappa(T-t)})$.

        With one argument, returns $V(0,t)$.
        """
        t = np.asarray(t, dtype=float)
        tau = t if T is None else np.asarray(T, dtype=float) - t
        return self.sigma**2 / (2.0 * self.kappa) * -np.expm1(-2.0 * self.kappa * tau)

    def simulate_x(self, times: ArrayLike, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Exact simulation of $X$ at increasing ``times`` (years, >= 0), starting from $X_0 = 0$.

        Returns an array of shape ``(len(times), n_paths)``.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times must be a 1-d non-decreasing array of non-negative values")
        steps = np.diff(times, prepend=0.0)
        decay = np.exp(-self.kappa * steps)
        sd = np.sqrt(self.variance(0.0, steps))
        z = rng.standard_normal((len(times), n_paths))
        x = np.empty_like(z)
        prev = np.zeros(n_paths)
        for i in range(len(times)):  # loop over time steps only; vectorised over paths
            prev = decay[i] * prev + sd[i] * z[i]
            x[i] = prev
        return x

    def spot(self, x: np.ndarray, times: ArrayLike, fwd: ArrayLike) -> np.ndarray:
        r"""$S_t = F(0,t)\exp(X_t - \tfrac12 V(0,t))$.

        ``x`` has shape ``(len(times), n_paths)``.
        """
        times = np.asarray(times, dtype=float)
        fwd = np.asarray(fwd, dtype=float)
        return fwd[:, None] * np.exp(x - 0.5 * self.variance(times)[:, None])

    def forward(self, x_t: np.ndarray, t: float, T: ArrayLike, fwd0_T: ArrayLike) -> np.ndarray:
        r"""Implied forward $F(t,T)$ given $X_t$ (see module docstring).

        Broadcasts over ``x_t`` and ``T``.
        """
        T = np.asarray(T, dtype=float)
        if np.any(t > T):
            raise ValueError("forward maturity T must be >= t")
        exponent = (
            np.exp(-self.kappa * (T - t)) * x_t + 0.5 * self.variance(t, T) - 0.5 * self.variance(T)
        )
        return np.asarray(fwd0_T, dtype=float) * np.exp(exponent)
