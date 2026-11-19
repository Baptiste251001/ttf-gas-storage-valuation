r"""Least-Squares Monte Carlo valuation of gas storage (Boogert & de Jong, 2008).

**Problem.** With inventory $v$ on the volume grid $\mathcal G$ and model state $X_{t_k}$, the value
of the storage satisfies

.. math::
    \mathcal V_k(v, X_{t_k}) = \max_{\Delta v\in\mathcal A_k(v)}
        \Big[\pi(\Delta v; S_{t_k}, D_k) + C_k(v + \Delta v, X_{t_k})\Big],
    \qquad C_k(v', x) = \mathbb E\big[\mathcal V_{k+1}(v', X_{t_{k+1}}) \mid X_{t_k} = x\big],

with the action set and cash flow $\pi$ of :mod:`gasstorage.storage.dp` (traded at the spot
$S_{t_k}$), all cash flows discounted to the valuation date, and the terminal condition
$\mathcal V_K$.

**Backward pass (regression paths, $n = 1..N_1$).** Starting from $\mathcal V_K$, for
$k = K-1, \dots, 0$ and for every grid level $v'$, regress the realised values on the basis
(:mod:`gasstorage.storage.basis`):

.. math::
    \hat\beta_k(v') = \arg\min_\beta \sum_n
        \Big(\mathcal V_{k+1}^{(n)}(v') - \phi(X^{(n)}_{t_k})^\top\beta\Big)^2,
    \qquad \hat C_k(v', x) = \phi(x)^\top\hat\beta_k(v').

On each path and grid level the action maximises cash plus $\hat C_k$ (linearly interpolated between
grid levels); the *realised* value $\pi(\Delta v^*) + \mathcal V_{k+1}^{(n)}(v+\Delta v^*)$ is
propagated backwards (Longstaff-Schwartz), not the fitted one.

**Forward pass (independent valuation paths, $n = 1..N_2$).** The policy defined by the
$\hat\beta_k$ is applied from $v_0$ on paths independent of the regression paths:

.. math::
    \hat V = \frac1{N_2}\sum_{n=1}^{N_2}\sum_{k} \pi\big(\Delta v_k^{(n)}; S^{(n)}_{t_k}, D_k\big),
    \qquad \mathrm{SE} = \frac{\hat s}{\sqrt{N_2}}.

Because the policy is feasible but sub-optimal and is evaluated on fresh paths, $\hat V$ is an
unbiased estimate of a **lower bound** of the true value. The in-sample estimate (mean of
$\mathcal V_0^{(n)}(v_0)$ on the regression paths) is reported as a diagnostic; it is usually
biased upwards because the policy has seen those paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.basis import state_basis
from gasstorage.storage.contract import ValuationInputs
from gasstorage.storage.dp import (
    MoveSearch,
    best_moves,
    grid_moves_by_days,
    interpolate,
    terminal_values,
    volume_levels,
)


@dataclass(frozen=True)
class Policy:
    """Exercise policy estimated by the backward pass: continuation coefficients per step/level."""

    betas: np.ndarray
    levels: np.ndarray
    basis: str
    degree: int


@dataclass(frozen=True)
class PolicyPaths:
    """A policy applied along given paths. ``cash[k]`` is the discounted cash flow of step k."""

    total: np.ndarray
    cash: np.ndarray
    moves: np.ndarray
    inventory: np.ndarray
    spot: np.ndarray


@dataclass(frozen=True)
class LSMCResult:
    """LSMC estimate (out-of-sample lower bound) with diagnostics."""

    value: float
    se: float
    in_sample_value: float
    path_values: np.ndarray
    inventory: np.ndarray
    moves: np.ndarray
    cash: np.ndarray
    spot: np.ndarray
    x: np.ndarray
    policy: Policy

    @property
    def betas(self) -> np.ndarray:
        return self.policy.betas

    @property
    def levels(self) -> np.ndarray:
        return self.policy.levels

    @property
    def basis(self) -> str:
        return self.policy.basis

    @property
    def degree(self) -> int:
        return self.policy.degree


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def lsmc_value(
    inp: ValuationInputs,
    model: OneFactorModel,
    n_paths_regression: int,
    n_paths_valuation: int,
    rng_regression: np.random.Generator,
    rng_valuation: np.random.Generator,
    n_levels: int = 101,
    basis: str = "polynomial",
    degree: int = 3,
    chunk: int = 2000,
) -> LSMCResult:
    """Value the storage by LSMC with independent regression and valuation path sets."""
    if rng_regression.bit_generator.state == rng_valuation.bit_generator.state:
        raise ValueError("regression and valuation paths must be independent (different seeds)")
    contract = inp.contract
    k_steps = len(inp.prices)
    levels = volume_levels(contract, n_levels)
    search = MoveSearch.build(contract, levels, float(np.max(inp.days)))
    d_end = float(inp.discount[-1])

    def basis_at(x_k: np.ndarray, k: int) -> np.ndarray:
        return state_basis(x_k, model, float(inp.times[k]), basis, degree)

    # ---- backward pass on regression paths ------------------------------------------------
    x_reg = model.simulate_x(inp.times, n_paths_regression, rng_regression)
    s_reg = model.spot(x_reg, inp.times, inp.prices)
    values = np.tile(terminal_values(contract, levels, d_end), (n_paths_regression, 1))
    grid_moves = grid_moves_by_days(contract, levels, inp.days, search)
    betas = np.empty((k_steps, degree + 1, n_levels))
    for k in range(k_steps - 1, -1, -1):
        phi = basis_at(x_reg[k], k)
        betas[k] = np.linalg.lstsq(phi, values, rcond=None)[0]
        fitted = phi @ betas[k]
        gm = grid_moves[float(inp.days[k])]
        new_values = np.empty_like(values)
        for sl in _chunks(n_paths_regression, chunk):
            _, _, cash, choice = gm.best(s_reg[k, sl], inp.discount[k], fitted[sl])
            new_values[sl] = cash + gm.realised(values[sl], choice)
        values = new_values
    in_sample = float(
        interpolate(
            values, np.full((len(values), 1), contract.initial_inventory), search.delta
        ).mean()
    )

    # ---- forward pass on independent valuation paths --------------------------------------
    policy = Policy(betas=betas, levels=levels, basis=basis, degree=degree)
    x_val = model.simulate_x(inp.times, n_paths_valuation, rng_valuation)
    paths = simulate_policy(policy, inp, model, x_val)
    total = paths.total
    se = float(total.std(ddof=1) / np.sqrt(n_paths_valuation)) if n_paths_valuation > 1 else np.nan
    return LSMCResult(
        value=float(total.mean()),
        se=se,
        in_sample_value=in_sample,
        path_values=total,
        inventory=paths.inventory,
        moves=paths.moves,
        cash=paths.cash,
        spot=paths.spot,
        x=x_val,
        policy=policy,
    )


def simulate_policy(
    policy: Policy, inp: ValuationInputs, model: OneFactorModel, x: np.ndarray
) -> PolicyPaths:
    """Apply ``policy`` from the initial inventory along model states ``x`` (shape ``(K, P)``)."""
    contract = inp.contract
    k_steps, n_paths = x.shape
    spot = model.spot(x, inp.times, inp.prices)
    search = MoveSearch.build(contract, policy.levels, float(np.max(inp.days)))
    v = np.full((n_paths, 1), contract.initial_inventory)
    inventory = np.empty((k_steps + 1, n_paths))
    moves = np.empty((k_steps, n_paths))
    cash = np.empty((k_steps, n_paths))
    inventory[0] = v[:, 0]
    for k in range(k_steps):
        phi = state_basis(x[k], model, float(inp.times[k]), policy.basis, policy.degree)
        fitted = phi @ policy.betas[k]
        _, dv, cf = best_moves(contract, v, spot[k], inp.discount[k], inp.days[k], fitted, search)
        cash[k], moves[k] = cf[:, 0], dv[:, 0]
        v = v + dv
        inventory[k + 1] = v[:, 0]
    total = cash.sum(axis=0)
    if contract.terminal_penalty is not None:
        total = total - float(inp.discount[-1]) * contract.terminal_penalty * np.abs(
            inventory[-1] - contract.terminal_inventory
        )
    return PolicyPaths(total=total, cash=cash, moves=moves, inventory=inventory, spot=spot)


def policy_map(
    result: LSMCResult,
    inp: ValuationInputs,
    model: OneFactorModel,
    k: int,
    x_values: np.ndarray,
    inventories: np.ndarray,
) -> np.ndarray:
    """Optimal move at step ``k`` for each model state in ``x_values`` and each inventory.

    Uses the fitted continuation of ``result``; returns ``dv`` of shape
    ``(len(x_values), len(inventories))``. The spot is $F(0,t_k)e^{x - V(0,t_k)/2}$.
    """
    contract = inp.contract
    x_values = np.asarray(x_values, dtype=float)
    inventories = np.asarray(inventories, dtype=float)
    phi = state_basis(x_values, model, float(inp.times[k]), result.basis, result.degree)
    fitted = phi @ result.betas[k]
    spot = model.spot(x_values[None, :], [inp.times[k]], [inp.prices[k]])[0]
    search = MoveSearch.build(contract, result.levels, float(np.max(inp.days)))
    v = np.broadcast_to(inventories, (len(x_values), len(inventories)))
    _, dv, _ = best_moves(contract, v, spot, inp.discount[k], inp.days[k], fitted, search)
    return dv
