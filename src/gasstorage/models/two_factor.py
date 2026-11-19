r"""Schwartz-Smith (2000) two-factor model fitted exactly to the forward curve.

.. math::
    \ln S_t = \ln F(0,t) + \chi_t + \xi_t - \tfrac12 V(0,t),\qquad
    d\chi_t = -\kappa\chi_t\,dt + \sigma_\chi dW^\chi_t,\qquad
    d\xi_t = \sigma_\xi dW^\xi_t,\qquad d\langle W^\chi, W^\xi\rangle_t = \rho\,dt,

with $\chi_0 = \xi_0 = 0$. The short-term factor $\chi$ mean-reverts, the long-term factor $\xi$
moves the whole curve. Its drift is absorbed by $F(0,t)$, so $\mathbb E[S_t] = F(0,t)$. With
$\tau = T - t$,

.. math::
    V(t,T) = \operatorname{Var}(\chi_T+\xi_T \mid \mathcal F_t)
        = \frac{\sigma_\chi^2}{2\kappa}(1-e^{-2\kappa\tau}) + \sigma_\xi^2\tau
          + \frac{2\rho\sigma_\chi\sigma_\xi}{\kappa}(1-e^{-\kappa\tau}),

.. math::
    F(t,T) = F(0,T)\exp\!\Big(e^{-\kappa\tau}\chi_t + \xi_t
        + \tfrac12 V(t,T) - \tfrac12 V(0,T)\Big),

a martingale because
$\operatorname{Var}(e^{-\kappa\tau}\chi_t + \xi_t) + V(t,T) = V(0,T)$.

**Exact simulation.** Over a step $\Delta$, $(\chi,\xi)$ moves by a Gaussian vector with
$\operatorname{Var}\varepsilon_\chi = \sigma_\chi^2(1-e^{-2\kappa\Delta})/(2\kappa)$,
$\operatorname{Var}\varepsilon_\xi = \sigma_\xi^2\Delta$ and
$\operatorname{Cov} = \rho\sigma_\chi\sigma_\xi(1-e^{-\kappa\Delta})/\kappa$, drawn by Cholesky
from two independent normal streams. The first stream drives $\chi$ exactly as $X$ is driven in
:class:`~gasstorage.models.one_factor.OneFactorModel`, so with $\sigma_\xi = 0$ both models produce
identical paths from the same seed.

**Regression basis.** Monomials of total degree $\le d$ in the standardised factors
$\chi_t/\operatorname{sd}(\chi_t)$ and $\xi_t/\operatorname{sd}(\xi_t)$.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TwoFactorModel:
    """Short-term OU factor and long-term Brownian factor; parameters annualised."""

    kappa: float
    sigma_chi: float
    sigma_xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if self.sigma_chi < 0 or self.sigma_xi < 0:
            raise ValueError("volatilities must be >= 0")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")

    # -- moments ------------------------------------------------------------------------------

    def _var_chi(self, tau: np.ndarray) -> np.ndarray:
        # same operation order as OneFactorModel.variance, so both models agree bit for bit
        return self.sigma_chi**2 / (2.0 * self.kappa) * -np.expm1(-2.0 * self.kappa * tau)

    def _var_xi(self, tau: np.ndarray) -> np.ndarray:
        return self.sigma_xi**2 * tau

    def _cov(self, tau: np.ndarray) -> np.ndarray:
        return self.rho * self.sigma_chi * self.sigma_xi * -np.expm1(-self.kappa * tau) / self.kappa

    def variance(self, t: ArrayLike, T: ArrayLike | None = None) -> np.ndarray:
        """$V(t,T)$ of $\\chi_T + \\xi_T$ given $\\mathcal F_t$; with one argument, $V(0,t)$."""
        t = np.asarray(t, dtype=float)
        tau = t if T is None else np.asarray(T, dtype=float) - t
        return self._var_chi(tau) + self._var_xi(tau) + 2.0 * self._cov(tau)

    # -- simulation and prices ----------------------------------------------------------------

    def simulate_state(
        self, times: ArrayLike, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Exact simulation of $(\\chi, \\xi)$ at increasing ``times``; shape ``(K, P, 2)``."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times must be a 1-d non-decreasing array of non-negative values")
        steps = np.diff(times, prepend=0.0)
        z1 = rng.standard_normal((len(times), n_paths))
        z2 = rng.standard_normal((len(times), n_paths))
        state = np.empty((len(times), n_paths, 2))
        chi = np.zeros(n_paths)
        xi = np.zeros(n_paths)
        for k, d in enumerate(steps):  # loop over time only; vectorised over paths
            v_chi, v_xi, c = self._var_chi(d), self._var_xi(d), self._cov(d)
            s_chi = np.sqrt(v_chi)
            a = c / s_chi if s_chi > 0 else 0.0
            b = np.sqrt(max(v_xi - a**2, 0.0))
            chi = np.exp(-self.kappa * d) * chi + s_chi * z1[k]
            xi = xi + a * z1[k] + b * z2[k]
            state[k, :, 0], state[k, :, 1] = chi, xi
        return state

    def spot(self, state: np.ndarray, times: ArrayLike, fwd: ArrayLike) -> np.ndarray:
        """$S_t = F(0,t)\\exp(\\chi_t + \\xi_t - V(0,t)/2)$ for ``state`` of shape ``(K, P, 2)``."""
        times = np.asarray(times, dtype=float)
        fwd = np.asarray(fwd, dtype=float)
        level = state[..., 0] + state[..., 1]
        return fwd[:, None] * np.exp(level - 0.5 * self.variance(times)[:, None])

    def forward_curve(
        self, state_t: np.ndarray, t: float, T: np.ndarray, fwd0: np.ndarray
    ) -> np.ndarray:
        """Forwards $F(t, T_j)$ for each path; ``state_t`` ``(P, 2)``, returns ``(P, len(T))``."""
        T = np.asarray(T, dtype=float)
        if np.any(t > T):
            raise ValueError("forward maturity T must be >= t")
        chi, xi = state_t[:, 0:1], state_t[:, 1:2]
        expo = (
            np.exp(-self.kappa * (T - t))[None, :] * chi
            + xi
            + 0.5 * self.variance(t, T)[None, :]
            - 0.5 * self.variance(T)[None, :]
        )
        return np.asarray(fwd0, dtype=float)[None, :] * np.exp(expo)

    def basis(self, state_k: np.ndarray, t: float, name: str, degree: int) -> np.ndarray:
        """Monomials of total degree <= ``degree`` in the standardised factors.

        A factor with zero variance carries no information, so its monomials are left out
        (with $\\sigma_\\xi = 0$ the basis is exactly the one-factor polynomial basis).
        """
        if name != "polynomial":
            raise ValueError("the two-factor model supports the polynomial basis only")
        sd_chi = float(np.sqrt(self._var_chi(np.asarray(t, dtype=float))))
        sd_xi = float(np.sqrt(self._var_xi(np.asarray(t, dtype=float))))
        n = len(state_k)
        u = state_k[:, 0] / sd_chi if sd_chi > 0 else np.zeros(n)
        v = state_k[:, 1] / sd_xi if sd_xi > 0 else np.zeros(n)
        powers = [(i, d - i) for d in range(degree + 1) for i in range(d, -1, -1)]
        if sd_xi == 0:
            powers = [(i, j) for i, j in powers if j == 0]
        if sd_chi == 0:
            powers = [(i, j) for i, j in powers if i == 0]
        # powers built like np.vander (as in the one-factor basis) for bitwise consistency
        pu = np.vander(u, degree + 1, increasing=True)
        pv = np.vander(v, degree + 1, increasing=True)
        return np.column_stack([pu[:, i] * pv[:, j] for i, j in powers])
