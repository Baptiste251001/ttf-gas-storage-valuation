r"""Regression basis functions for the LSMC continuation value.

The continuation value at step $k$ is regressed on functions of the model state $X_{t_k}$:

* ``polynomial`` (default): $\phi_j(z) = z^j$, $j = 0..d$, with the standardised state
  $z = X_{t_k}/\sqrt{V(0,t_k)}$. The scaling is the model's unconditional standard deviation, a
  deterministic number, so regression and valuation paths use exactly the same basis.
* ``laguerre``: weighted Laguerre polynomials (Longstaff & Schwartz, 2001)
  $\phi_j(s) = e^{-s/2} L_j(s)$ of the normalised spot
  $s = S_{t_k}/F(0,t_k) = e^{X_{t_k} - V(0,t_k)/2}$,
  with $L_0 = 1$, $L_1(s) = 1 - s$, $L_2(s) = (s^2 - 4s + 2)/2$, ...

Both bases contain a constant, so a deterministic continuation is fitted exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_laguerre

from gasstorage.models.one_factor import OneFactorModel


def polynomial_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Columns ``[1, z, z^2, ..., z^degree]``."""
    return np.vander(np.asarray(z, dtype=float), degree + 1, increasing=True)


def laguerre_basis(s: np.ndarray, degree: int) -> np.ndarray:
    """Columns ``exp(-s/2) L_j(s)`` for ``j = 0..degree``."""
    s = np.asarray(s, dtype=float)
    weight = np.exp(-0.5 * s)
    return np.column_stack([weight * eval_laguerre(j, s) for j in range(degree + 1)])


def state_basis(
    x: np.ndarray, model: OneFactorModel, t: float, name: str, degree: int
) -> np.ndarray:
    """Basis evaluated at the model state ``x`` (shape ``(P,)``) at time ``t`` (years)."""
    var = float(model.variance(t))
    if name == "polynomial":
        z = np.asarray(x) / np.sqrt(var) if var > 0 else np.zeros_like(x, dtype=float)
        return polynomial_basis(z, degree)
    if name == "laguerre":
        return laguerre_basis(np.exp(np.asarray(x) - 0.5 * var), degree)
    raise ValueError(f"unknown basis {name!r}")
