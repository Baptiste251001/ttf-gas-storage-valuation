r"""Dynamic programming on a volume grid, vectorised over price scenarios.

Volume grid $\mathcal G = \{0, \Delta, 2\Delta, \dots, \bar V\}$ with $\Delta = \bar V/(G-1)$.
Backward recursion, for every grid level $v$ and step $k$:

.. math::
    \mathcal V_K(v) = -D_{K-1}\,p\,|v - v^{\text{end}}|, \qquad
    \mathcal V_k(v) = \max_{\Delta v \in \mathcal A_k(v)}
        \Big[\pi(\Delta v; P_k, D_k) + \widehat{\mathcal V}_{k+1}(v + \Delta v)\Big],

where $\widehat{\mathcal V}_{k+1}$ is the **linear interpolation** of $\mathcal V_{k+1}$ between
grid levels and the action set $\mathcal A_k(v)$ contains: holding ($\Delta v = 0$), full injection
$\Delta v = \min(q^{\text{inj}}d_k\phi^{\text{inj}}(v/\bar V),\ \bar V - v)$, full withdrawal
$\Delta v = -\min(q^{\text{wdr}}d_k\phi^{\text{wdr}}(v/\bar V),\ v)$, and every move to a grid
level within these limits. Because cash flows are linear in $\Delta v$, the optimum against a
piecewise-linear continuation is attained at one of these points.

A hard terminal constraint is imposed through a large penalty
$p = 10^4$ EUR/MWh, far above any price, so the optimal policy always ends at $v^{\text{end}}$
(which is a grid level). Prices may carry a leading scenario axis: the same code solves the
deterministic intrinsic problem, the re-optimisations of the rolling intrinsic strategy, and
evaluates the action choice of the LSMC policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gasstorage.storage.contract import StorageContract

HARD_PENALTY = 1e4  # EUR/MWh, enforces a hard terminal constraint


def volume_levels(contract: StorageContract, n_levels: int) -> np.ndarray:
    """Uniform volume grid from 0 to the working volume."""
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    return np.linspace(0.0, contract.working_volume, n_levels)


def terminal_values(
    contract: StorageContract, levels: np.ndarray, discount_end: float
) -> np.ndarray:
    """Terminal value $\\mathcal V_K$ on the grid (penalty for missing the terminal inventory)."""
    pen = HARD_PENALTY if contract.terminal_penalty is None else contract.terminal_penalty
    return -discount_end * pen * np.abs(levels - contract.terminal_inventory)


def interpolate(values: np.ndarray, targets: np.ndarray, delta: float) -> np.ndarray:
    """Per-scenario linear interpolation of grid values.

    ``values`` has shape ``(P, G)``; ``targets`` has shape ``(P, ...)`` in MWh.
    """
    p, g = values.shape
    pos = np.clip(targets / delta, 0.0, g - 1.0).reshape(p, -1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), g - 2)
    w = pos - i0
    v0 = np.take_along_axis(values, i0, axis=1)
    v1 = np.take_along_axis(values, i0 + 1, axis=1)
    return (v0 + w * (v1 - v0)).reshape(targets.shape)


@dataclass(frozen=True)
class MoveSearch:
    """Precomputed band of grid offsets that covers every feasible move on the grid.

    With ``extremes=True`` (default) the action set also contains the full-rate moves and holding,
    which may leave the grid (continuation values are then interpolated). With ``extremes=False``
    only moves to grid levels are allowed: the *grid-restricted problem*, whose states stay on the
    grid, used for the dual upper bound.
    """

    delta: float
    offsets: np.ndarray
    extremes: bool = True

    @classmethod
    def build(
        cls, contract: StorageContract, levels: np.ndarray, max_days: float, extremes: bool = True
    ) -> MoveSearch:
        delta = float(levels[1] - levels[0])
        n_inj = math.ceil(contract.max_injection_per_day * max_days / delta) + 1
        n_wdr = math.ceil(contract.max_withdrawal_per_day * max_days / delta) + 1
        return cls(delta=delta, offsets=np.arange(-n_wdr, n_inj + 1) * delta, extremes=extremes)


def best_moves(
    contract: StorageContract,
    v: np.ndarray,
    price: np.ndarray,
    discount: float,
    days: float,
    value_next: np.ndarray,
    search: MoveSearch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal move from inventories ``v`` (shape ``(P, M)``) given next-step grid values.

    ``price`` has shape ``(P,)`` and ``value_next`` shape ``(P, G)``. Returns the best total
    (cash + interpolated continuation), the move $\\Delta v$ and its cash flow, each ``(P, M)``.
    """
    vol = contract.working_volume
    tol = 1e-9 * vol
    inj = np.minimum(contract.max_injection(v, days), vol - v)
    wdr = np.minimum(contract.max_withdrawal(v, days), v)
    base = np.floor(v / search.delta + 1e-9) * search.delta
    parts = [base[..., None] + search.offsets]
    if search.extremes:
        parts += [(v + inj)[..., None], (v - wdr)[..., None], v[..., None]]
    targets = np.concatenate(parts, axis=-1)
    dv = targets - v[..., None]
    feasible = (
        (dv <= inj[..., None] + tol)
        & (dv >= -wdr[..., None] - tol)
        & (targets >= -tol)
        & (targets <= vol + tol)
    )
    cont = interpolate(value_next, np.clip(targets, 0.0, vol), search.delta)
    cash = contract.cash_flow(dv, price[:, None, None], discount)
    total = np.where(feasible, cash + cont, -np.inf)
    best = np.argmax(total, axis=-1)[..., None]
    pick = lambda a: np.take_along_axis(a, best, axis=-1)[..., 0]  # noqa: E731
    return pick(total), pick(dv), pick(cash)


@dataclass(frozen=True)
class GridMoves:
    """Path-independent action set for states lying on the grid (backward passes).

    For grid states the candidate moves, their feasibility and the interpolation weights of
    their targets depend only on the grid level and the step length, so they are computed once.
    Per scenario only the cash flow, linear in the price, and a gather of continuation values
    remain. Candidates and tie-breaking are identical to :func:`best_moves`.
    """

    dv: np.ndarray
    feasible: np.ndarray
    i0: np.ndarray
    w: np.ndarray
    inj_volume: np.ndarray
    wdr_volume: np.ndarray
    contract: StorageContract

    @classmethod
    def build(
        cls, contract: StorageContract, levels: np.ndarray, days: float, search: MoveSearch
    ) -> GridMoves:
        vol = contract.working_volume
        tol = 1e-9 * vol
        v = levels
        inj = np.minimum(contract.max_injection(v, days), vol - v)
        wdr = np.minimum(contract.max_withdrawal(v, days), v)
        base = np.floor(v / search.delta + 1e-9) * search.delta
        parts = [base[:, None] + search.offsets]
        if search.extremes:
            parts += [(v + inj)[:, None], (v - wdr)[:, None], v[:, None]]
        targets = np.concatenate(parts, axis=1)
        dv = targets - v[:, None]
        feasible = (
            (dv <= inj[:, None] + tol)
            & (dv >= -wdr[:, None] - tol)
            & (targets >= -tol)
            & (targets <= vol + tol)
        )
        g = len(levels)
        pos = np.clip(np.clip(targets, 0.0, vol) / search.delta, 0.0, g - 1.0)
        i0 = np.minimum(np.floor(pos).astype(np.int64), g - 2)
        return cls(
            dv=dv,
            feasible=feasible,
            i0=i0,
            w=pos - i0,
            inj_volume=np.maximum(dv, 0.0),
            wdr_volume=np.maximum(-dv, 0.0),
            contract=contract,
        )

    def best(
        self, price: np.ndarray, discount: float, value_next: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Best move from every grid level for each scenario.

        Returns total, move, cash flow (each ``(P, G)``) and the chosen candidate index.
        """
        c = self.contract
        v0 = value_next[:, self.i0]
        cont = v0 + self.w * (value_next[:, self.i0 + 1] - v0)
        buy = (price * (1.0 + c.fuel_loss) + c.injection_cost)[:, None, None]
        sell = (price - c.withdrawal_cost)[:, None, None]
        cash = discount * (-buy * self.inj_volume + sell * self.wdr_volume)
        total = np.where(self.feasible, cash + cont, -np.inf)
        best = np.argmax(total, axis=-1)[..., None]
        pick = lambda a: np.take_along_axis(a, best, axis=-1)[..., 0]  # noqa: E731
        rows = np.arange(self.dv.shape[0])
        return pick(total), self.dv[rows, best[..., 0]], pick(cash), best[..., 0]

    def realised(self, values: np.ndarray, choice: np.ndarray) -> np.ndarray:
        """Interpolated ``values`` (``(P, G)``) at the targets of the chosen candidates."""
        rows = np.arange(self.dv.shape[0])
        i0 = self.i0[rows, choice]
        w = self.w[rows, choice]
        v0 = np.take_along_axis(values, i0, axis=1)
        return v0 + w * (np.take_along_axis(values, i0 + 1, axis=1) - v0)


def grid_moves_by_days(
    contract: StorageContract, levels: np.ndarray, days: np.ndarray, search: MoveSearch
) -> dict[float, GridMoves]:
    """One :class:`GridMoves` per distinct step length."""
    return {float(d): GridMoves.build(contract, levels, float(d), search) for d in np.unique(days)}


def backward_values(
    contract: StorageContract,
    prices: np.ndarray,
    days: np.ndarray,
    discount: np.ndarray,
    levels: np.ndarray,
    discount_end: float | None = None,
) -> np.ndarray:
    """Grid values $\\mathcal V_k$ for ``prices`` of shape ``(P, K)``; returns ``(K + 1, P, G)``."""
    p, k = prices.shape
    search = MoveSearch.build(contract, levels, float(np.max(days)))
    d_end = float(discount[-1]) if discount_end is None else discount_end
    values = np.empty((k + 1, p, len(levels)))
    values[k] = terminal_values(contract, levels, d_end)[None, :]
    moves = grid_moves_by_days(contract, levels, days, search)
    for step in range(k - 1, -1, -1):
        values[step] = moves[float(days[step])].best(
            prices[:, step], discount[step], values[step + 1]
        )[0]
    return values


@dataclass(frozen=True)
class DPResult:
    """Grid DP solution. With 1-d prices, ``value`` is a float and paths are squeezed."""

    value: float | np.ndarray
    levels: np.ndarray
    values: np.ndarray
    inventory: np.ndarray
    moves: np.ndarray
    cash: np.ndarray


def solve_dp(
    contract: StorageContract,
    prices: np.ndarray,
    days: np.ndarray,
    discount: np.ndarray,
    n_levels: int = 101,
) -> DPResult:
    """Backward DP then forward policy simulation from the initial inventory.

    The reported value is the realised discounted cash flow of the forward pass (plus the
    terminal penalty if the terminal condition is soft).
    """
    prices = np.asarray(prices, dtype=float)
    squeeze = prices.ndim == 1
    prices2 = prices[None, :] if squeeze else prices
    days = np.asarray(days, dtype=float)
    discount = np.asarray(discount, dtype=float)
    p, k = prices2.shape
    levels = volume_levels(contract, n_levels)
    values = backward_values(contract, prices2, days, discount, levels)
    search = MoveSearch.build(contract, levels, float(np.max(days)))

    v = np.full((p, 1), contract.initial_inventory)
    inventory = np.empty((k + 1, p))
    moves = np.empty((k, p))
    cash = np.empty((k, p))
    inventory[0] = v[:, 0]
    for step in range(k):
        _, dv, cf = best_moves(
            contract, v, prices2[:, step], discount[step], days[step], values[step + 1], search
        )
        moves[step], cash[step] = dv[:, 0], cf[:, 0]
        v = v + dv
        inventory[step + 1] = v[:, 0]
    total = cash.sum(axis=0)
    if contract.terminal_penalty is not None:
        total -= (
            discount[-1]
            * contract.terminal_penalty
            * np.abs(inventory[-1] - contract.terminal_inventory)
        )
    if squeeze:
        return DPResult(
            float(total[0]), levels, values[:, 0], inventory[:, 0], moves[:, 0], cash[:, 0]
        )
    return DPResult(total, levels, values, inventory, moves, cash)
