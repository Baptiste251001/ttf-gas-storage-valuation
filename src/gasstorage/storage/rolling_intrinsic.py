r"""Rolling intrinsic strategy along simulated price paths.

At each re-optimisation date $t_k$ (by default the first decision date of every month) the
intrinsic problem is re-solved from the current inventory $v_k$ against the model-implied forward
curve of that path,

.. math::
    F(t_k, t_j) = F(0,t_j)\exp\!\Big(e^{-\kappa(t_j-t_k)}X_{t_k}
        + \tfrac12 V(t_k,t_j) - \tfrac12 V(0,t_j)\Big), \qquad j \ge k,

and the resulting schedule is followed until the next re-optimisation. Volumes are settled at the
spot price $S_{t_j}$. The textbook version locks each schedule with forward trades; since forwards
are martingales those hedges have zero expected P&L, so the expected discounted cash flow is the
same and equals the rolling intrinsic value

.. math::
    V^{\text{RI}} = \mathbb E\Big[\sum_k \pi(\Delta v_k; S_{t_k}, D_k)\Big]
    \approx \frac1N \sum_{n=1}^{N} \sum_k \pi(\Delta v_k^{(n)}; S^{(n)}_{t_k}, D_k),

reported with its Monte Carlo standard error. Each schedule is feasible and each re-optimisation
can only raise the planned value, so $V^{\text{RI}} \ge V^{\text{int}}$ (in expectation): it is a
lower bound of the full option value that captures *forward-curve* moves but not the optionality of
reacting to the spot within a month.

Forwards are computed per decision date (daily delivery), not per monthly contract; monthly
contracts are used for hedging in :mod:`gasstorage.risk.delta_hedge`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.contract import ValuationInputs
from gasstorage.storage.dp import MoveSearch, backward_values, best_moves, volume_levels


@dataclass(frozen=True)
class RollingIntrinsicResult:
    """Monte Carlo estimate of the rolling intrinsic value."""

    value: float
    se: float
    path_values: np.ndarray
    inventory: np.ndarray
    n_reoptimisations: int


def monthly_reoptimisation(dates) -> np.ndarray:
    """Boolean mask: first decision date of each calendar month (and the first date)."""
    months = dates.year * 12 + dates.month
    return np.r_[True, months[1:] != months[:-1]]


def rolling_intrinsic(
    inp: ValuationInputs,
    model: OneFactorModel,
    n_paths: int,
    rng: np.random.Generator,
    n_levels: int = 51,
    reoptimise: np.ndarray | None = None,
) -> RollingIntrinsicResult:
    """Simulate the rolling intrinsic strategy on ``n_paths`` paths of the one-factor model."""
    contract = inp.contract
    k_steps = len(inp.prices)
    mask = monthly_reoptimisation(inp.dates) if reoptimise is None else np.asarray(reoptimise)
    if not mask[0]:
        raise ValueError("the first decision date must be a re-optimisation date")
    x = model.simulate_x(inp.times, n_paths, rng)
    spot = model.spot(x, inp.times, inp.prices)
    levels = volume_levels(contract, n_levels)
    search = MoveSearch.build(contract, levels, float(np.max(inp.days)))
    d_end = float(inp.discount[-1])

    v = np.full((n_paths, 1), contract.initial_inventory)
    inventory = np.empty((k_steps + 1, n_paths))
    inventory[0] = v[:, 0]
    total = np.zeros(n_paths)
    plan_start, plan_prices, plan_values = 0, None, None
    for k in range(k_steps):
        if mask[k]:
            plan_start = k
            plan_prices = model.forward(
                x[k][:, None], inp.times[k], inp.times[None, k:], inp.prices[None, k:]
            )
            plan_values = backward_values(
                contract, plan_prices, inp.days[k:], inp.discount[k:], levels, d_end
            )
        j = k - plan_start
        _, dv, _ = best_moves(
            contract, v, plan_prices[:, j], inp.discount[k], inp.days[k], plan_values[j + 1], search
        )
        total += contract.cash_flow(dv[:, 0], spot[k], inp.discount[k])
        v = v + dv
        inventory[k + 1] = v[:, 0]
    if contract.terminal_penalty is not None:
        total -= (
            d_end * contract.terminal_penalty * np.abs(inventory[-1] - contract.terminal_inventory)
        )
    se = float(total.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")
    return RollingIntrinsicResult(float(total.mean()), se, total, inventory, int(mask.sum()))
