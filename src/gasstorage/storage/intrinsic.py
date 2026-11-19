r"""Intrinsic value: optimal storage schedule against today's forward curve (cvxpy + HiGHS).

The intrinsic value locks in today's forward prices $F(0,t_k)$ and ignores all optionality:

.. math::
    V^{\text{int}} = \max_{i,w \ge 0}\ \sum_{k=0}^{K-1} D_k\Big[\big(F_k - c^{\text{wdr}}\big)w_k
        - \big(F_k(1+\lambda) + c^{\text{inj}}\big)i_k\Big] - D_{K-1}\,p\,(s^+ + s^-)

subject to $v_0$ given, $v_{k+1} = v_k + i_k - w_k$, $0 \le v_k \le \bar V$, the rate limits, and
the terminal condition: $v_K = v^{\text{end}}$ (hard) or $v_K - v^{\text{end}} = s^+ - s^-$,
$s^\pm \ge 0$ (soft, penalty $p$).

**Ratchets.** Rate limits that depend on the fill level make the feasible set non-convex, so two
formulations are provided:

* ``ratchets=False``: a **linear program** with the nominal (unratcheted) rates. It is an upper
  bound of the ratcheted value and the exact value of a contract without ratchets.
* ``ratchets=True``: a **mixed-integer program**. For each step and each ratchet schedule, binary
  variables $z_{k,j}$ select the tier of the start-of-step fill level:

  .. math::
     \sum_j z_{k,j} = 1,\quad
     \bar V \sum_j a_j z_{k,j} \le v_k \le \bar V \sum_j (b_j - \varepsilon_j) z_{k,j},\quad
     i_k \le q^{\text{inj}} d_k \sum_j \phi_j z_{k,j},

  with $\varepsilon_j = 10^{-6}$ for every tier but the last, so that tiers are half-open
  $[a_j, b_j)$ exactly as in the dynamic programme of :mod:`gasstorage.storage.dp`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from gasstorage.config import RatchetTier
from gasstorage.storage.contract import StorageContract

_EPS_FILL = 1e-6


@dataclass(frozen=True)
class IntrinsicResult:
    """Optimal intrinsic schedule. Arrays have length K (flows) or K + 1 (inventory)."""

    value: float
    injection: np.ndarray
    withdrawal: np.ndarray
    inventory: np.ndarray
    method: str


def _tier_constraints(
    tiers: tuple[RatchetTier, ...],
    flow: cp.Variable,
    v_start: cp.Expression,
    rate: np.ndarray,
    volume: float,
) -> list[cp.Constraint]:
    if len(tiers) == 1:
        return [flow <= rate * tiers[0].factor]
    n = len(tiers)
    lower = np.array([t.lower for t in tiers]) * volume
    upper = np.array([t.upper - (_EPS_FILL if j < n - 1 else 0.0) for j, t in enumerate(tiers)])
    factor = np.array([t.factor for t in tiers])
    z = cp.Variable((flow.shape[0], n), boolean=True)
    return [
        cp.sum(z, axis=1) == 1,
        v_start >= z @ lower,
        v_start <= z @ (upper * volume),
        flow <= cp.multiply(rate, z @ factor),
    ]


def solve_lp(
    contract: StorageContract,
    prices: np.ndarray,
    days: np.ndarray,
    discount: np.ndarray,
    ratchets: bool = False,
) -> IntrinsicResult:
    """Solve the intrinsic problem as an LP (``ratchets=False``) or a MILP (``ratchets=True``)."""
    c = contract if ratchets else contract.without_ratchets()
    prices, days, discount = (np.asarray(a, dtype=float) for a in (prices, days, discount))
    k = len(prices)
    vol = c.working_volume
    inj = cp.Variable(k, nonneg=True)
    wdr = cp.Variable(k, nonneg=True)
    v = cp.Variable(k + 1)
    cons = [v[0] == c.initial_inventory, v[1:] == v[:-1] + inj - wdr, v >= 0, v <= vol]
    cons += _tier_constraints(
        c.injection_ratchets, inj, v[:-1], c.max_injection_per_day * days, vol
    )
    cons += _tier_constraints(
        c.withdrawal_ratchets, wdr, v[:-1], c.max_withdrawal_per_day * days, vol
    )
    buy = discount * (prices * (1.0 + c.fuel_loss) + c.injection_cost)
    sell = discount * (prices - c.withdrawal_cost)
    objective = sell @ wdr - buy @ inj
    if c.terminal_penalty is None:
        cons.append(v[k] == c.terminal_inventory)
    else:
        slack = cp.Variable(2, nonneg=True)
        cons.append(v[k] - c.terminal_inventory == slack[0] - slack[1])
        objective -= discount[-1] * c.terminal_penalty * cp.sum(slack)
    problem = cp.Problem(cp.Maximize(objective), cons)
    problem.solve(solver=cp.HIGHS)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"intrinsic problem not solved: {problem.status}")
    is_mip = ratchets and c.has_ratchets
    return IntrinsicResult(
        value=float(problem.value),
        injection=np.asarray(inj.value),
        withdrawal=np.asarray(wdr.value),
        inventory=np.asarray(v.value),
        method="milp" if is_mip else "lp",
    )
