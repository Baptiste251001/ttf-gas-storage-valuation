r"""Dual (information-relaxation) upper bound for the grid-restricted storage problem.

**Problem bounded.** Inventory restricted to the volume grid and moves to grid levels within the
rate limits (``moves="grid"``). For this problem the LSMC policy evaluated on independent paths is
a lower bound and this module computes an upper bound, so the two bracket its optimal value.

**Duality** (Rogers, 2002; Haugh & Kogan, 2004; Andersen & Broadie, 2004; Brown, Smith & Sun, 2010).
Let $p_k(v')$ be penalties with $\mathbb E[p_k(v') \mid \mathcal F_{t_k}] = 0$ for every target
level $v'$. Any non-anticipative policy chooses $v_{k+1}$ at $t_k$, so
$\mathbb E[\sum_k p_k(v_{k+1})] = 0$ and

.. math::
    V^* = \sup_\pi \mathbb E\Big[\sum_k \pi_k\Big]
        = \sup_\pi \mathbb E\Big[\sum_k \big(\pi_k - p_k(v_{k+1})\big)\Big]
        \le \mathbb E\Big[\max_{v_1,\dots,v_K} \sum_k \big(\pi_k(v_k \to v_{k+1}; S_{t_k})
            - p_k(v_{k+1})\big)\Big],

where the inner maximum is a **deterministic** dynamic programme on each path (perfect foresight
of the prices, corrected by the penalties). With the exact value function as penalty basis the
bound is tight.

**Penalty.** From the LSMC backward pass, $\hat V_{k}(v', x) = \phi(x)^\top\hat\gamma_k(v')$
regresses the realised values on the basis at $t_k$. The penalty is its innovation

.. math::
    p_k(v') = \big(\phi(X_{t_{k+1}}) - \mathbb E[\phi(X_{t_{k+1}}) \mid X_{t_k}]\big)^\top
              \hat\gamma_{k+1}(v').

With the polynomial basis in $z = X/c$, $c = \sqrt{V(0,t_{k+1})}$, and the Gaussian transition
$X_{t_{k+1}} \mid X_{t_k} \sim \mathcal N(e^{-\kappa\Delta}X_{t_k}, V(t_k,t_{k+1}))$, the
conditional expectation is analytic: $\mathbb E[(\mu + sZ)^j]
= \sum_{m \text{ even}} \binom{j}{m}\mu^{j-m}s^m(m-1)!!$. No nested simulation is needed,
which is the main cost of the original Andersen-Broadie construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.contract import ValuationInputs
from gasstorage.storage.dp import MoveSearch, grid_moves_by_days, terminal_values
from gasstorage.storage.lsmc import Policy


def _double_factorial(n: int) -> int:
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def normal_raw_moments(mu: np.ndarray, sd: float, degree: int) -> np.ndarray:
    """$\\mathbb E[(\\mu + sZ)^j]$ for $j = 0..d$ and $Z \\sim \\mathcal N(0,1)$.

    Returns an array of shape ``(n, d + 1)``.
    """
    mu = np.asarray(mu, dtype=float)
    out = np.zeros((len(mu), degree + 1))
    for j in range(degree + 1):
        for m in range(0, j + 1, 2):
            out[:, j] += comb(j, m) * mu ** (j - m) * sd**m * _double_factorial(m - 1)
    return out


def conditional_basis_mean(
    model: OneFactorModel, x0: np.ndarray, t0: float, t1: float, degree: int
) -> np.ndarray:
    """$\\mathbb E[\\phi(X_{t_1}) \\mid X_{t_0} = x_0]$.

    For the polynomial basis in $X/\\sqrt{V(0,t_1)}$ of :mod:`gasstorage.storage.basis`.
    """
    c = float(np.sqrt(model.variance(t1)))
    if c == 0.0:
        out = np.zeros((len(x0), degree + 1))
        out[:, 0] = 1.0
        return out
    mean = np.exp(-model.kappa * (t1 - t0)) * np.asarray(x0, dtype=float) / c
    sd = float(np.sqrt(model.variance(t0, t1))) / c
    return normal_raw_moments(mean, sd, degree)


@dataclass(frozen=True)
class DualResult:
    """Upper bound with standard error, and penalty diagnostics per step and level."""

    value: float
    se: float
    path_values: np.ndarray
    mean_penalty: np.ndarray
    penalty_se: np.ndarray


def dual_upper_bound(
    inp: ValuationInputs,
    model: OneFactorModel,
    policy: Policy,
    n_paths: int,
    rng: np.random.Generator,
) -> DualResult:
    """Upper bound of the grid-restricted problem from the regression martingale of ``policy``."""
    if not isinstance(model, OneFactorModel):
        raise TypeError("the analytic penalty is implemented for the one-factor model")
    if policy.moves != "grid" or policy.basis != "polynomial" or policy.value_betas is None:
        raise ValueError("needs an LSMC policy with moves='grid' and the polynomial basis")
    contract = inp.contract
    levels = policy.levels
    k_steps = len(inp.prices)
    i0 = int(np.argmin(np.abs(levels - contract.initial_inventory)))
    if not np.isclose(levels[i0], contract.initial_inventory):
        raise ValueError("the initial inventory must be a grid level")

    x = model.simulate_state(inp.times, n_paths, rng)
    spot = model.spot(x, inp.times, inp.prices)
    search = MoveSearch.build(contract, levels, float(np.max(inp.days)), extremes=False)
    moves = grid_moves_by_days(contract, levels, inp.days, search)

    w = np.tile(terminal_values(contract, levels, float(inp.discount[-1])), (n_paths, 1))
    mean_pen = np.zeros((k_steps, len(levels)))
    se_pen = np.zeros((k_steps, len(levels)))
    for k in range(k_steps - 1, -1, -1):
        if k + 1 < k_steps:
            t0, t1 = float(inp.times[k]), float(inp.times[k + 1])
            phi_next = model.basis(x[k + 1], t1, "polynomial", policy.degree)
            phi_mean = conditional_basis_mean(model, x[k], t0, t1, policy.degree)
            penalty = (phi_next - phi_mean) @ policy.value_betas[k + 1]
            mean_pen[k] = penalty.mean(axis=0)
            se_pen[k] = penalty.std(axis=0, ddof=1) / np.sqrt(n_paths)
        else:
            penalty = 0.0
        w = moves[float(inp.days[k])].best(spot[k], inp.discount[k], w - penalty)[0]
    values = w[:, i0]
    se = float(values.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")
    return DualResult(float(values.mean()), se, values, mean_pen, se_pen)
