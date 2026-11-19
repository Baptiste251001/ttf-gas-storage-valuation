r"""Delta hedge of the storage with monthly forwards, and hedged vs unhedged P&L.

**Monthly forward.** The forward for delivery month $m$ is the day-weighted average of the
model forwards of its decision dates, $F_m(\tau) = \sum_{k\in m} \omega_k F(\tau, t_k)$ with
$\omega_k = d_k/\sum_{j\in m} d_j$. It trades until the first decision date $t_{k_m}$ of the
month; afterwards the remaining exposure of that month is to the spot.

**Hedge ratio.** By the envelope theorem the policy can be held fixed when differentiating the
optimal value. In the one-factor model $S_{t_k} \propto F(\tau, t_k)$ along each path, so a
relative bump of $F_m(\tau)$ moves the spot of month $m$ proportionally and

.. math::
    \Delta_m(\tau) = \frac{\partial V_\tau}{\partial F_m(\tau)}
        = \frac{1}{F_m(\tau)}\,\mathbb E_\tau\Big[\sum_{k\in m} D_k
          \big(w_k - (1+\lambda) i_k\big) S_{t_k}\Big].

The conditional expectation is estimated by regressing the realised bracket on a tensor
polynomial basis of $(X_\tau, v_\tau/\bar V)$ on a set of **fit paths**, independent of the
**test paths** on which the hedge is evaluated. The position is $q_m(\tau) = -\Delta_m(\tau)/D_m$
forwards (MWh), because a forward settles at delivery and its P&L is discounted by $D_m$.

**Hedge P&L and costs.** Between rebalancing dates $\tau_j < \tau_{j+1}$,
$\text{P\&L}_m = q_m(\tau_j)\,D_m\,\big(F_m(\tau_{j+1}) - F_m(\tau_j)\big)$; each trade costs
half the bid-ask spread, $\tfrac12 b\,|q_m(\tau_j) - q_m(\tau_{j-1})|\,D(\tau_j)$;
expiring positions settle financially at no cost. Rebalancing dates: the valuation date, the
first day of each month before delivery starts, then the first decision date of each month
(``monthly``) or every decision date (``weekly``).

Residual risk comes from discrete rebalancing, regression error and the spot exposure inside each
delivery month, which monthly forwards cannot hedge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.basis import polynomial_basis
from gasstorage.storage.contract import ValuationInputs
from gasstorage.storage.lsmc import Policy, simulate_policy
from gasstorage.timegrid import year_fraction


@dataclass(frozen=True)
class HedgeResult:
    """Per-path discounted P&L on the test paths (EUR)."""

    unhedged: np.ndarray
    hedged: np.ndarray
    hedge_pnl: np.ndarray
    costs: np.ndarray
    hedge_dates: pd.DatetimeIndex
    initial_delta: pd.Series
    n_trades: int


def pnl_statistics(pnl: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Mean, standard deviation, 95 % VaR and expected shortfall (losses as positive numbers)."""
    rows = {}
    for name, x in pnl.items():
        q = np.quantile(x, 0.05)
        rows[name] = {
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "VaR 95 %": -q,
            "ES 95 %": -x[x <= q].mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _hedge_dates(inp: ValuationInputs, rebalance: str) -> pd.DatetimeIndex:
    start = inp.dates[0]
    pre = pd.date_range(
        pd.Timestamp(inp.valuation_date) + pd.offsets.MonthBegin(1), start, freq="MS"
    )
    pre = pre[pre < start]
    if rebalance == "weekly":
        during = inp.dates
    elif rebalance == "monthly":
        months = inp.dates.year * 12 + inp.dates.month
        during = inp.dates[np.r_[True, months[1:] != months[:-1]]]
    else:
        raise ValueError(f"unknown rebalance {rebalance!r}")
    return pd.DatetimeIndex([pd.Timestamp(inp.valuation_date)]).append(pre).append(during)


def _tensor_basis(z: np.ndarray, u: np.ndarray, deg_z: int, deg_u: int) -> np.ndarray:
    bz, bu = polynomial_basis(z, deg_z), polynomial_basis(u, deg_u)
    return (bz[:, :, None] * bu[:, None, :]).reshape(len(z), -1)


def delta_hedge(
    inp: ValuationInputs,
    model: OneFactorModel,
    policy: Policy,
    n_paths_fit: int,
    n_paths_test: int,
    rng_fit: np.random.Generator,
    rng_test: np.random.Generator,
    rebalance: str = "monthly",
    bid_ask: float = 0.0,
    degree_z: int = 3,
    degree_v: int = 2,
) -> HedgeResult:
    """Simulate the storage policy with and without a monthly-forward delta hedge."""
    contract = inp.contract
    hedge_dates = _hedge_dates(inp, rebalance)
    tau_h = year_fraction(inp.valuation_date, hedge_dates)
    times_all = np.unique(np.concatenate([tau_h, inp.times]))
    idx_h = np.searchsorted(times_all, tau_h)
    idx_k = np.searchsorted(times_all, inp.times)
    rate = -np.log(inp.discount[0]) / inp.times[0] if inp.times[0] > 0 else 0.0
    disc_h = np.exp(-rate * tau_h)

    step_month = inp.dates.to_period("M")
    months = pd.PeriodIndex(step_month.unique())
    steps = [np.flatnonzero(step_month == m) for m in months]
    weights = [inp.days[s] / inp.days[s].sum() for s in steps]
    first = np.array([s[0] for s in steps])
    d_m = inp.discount[first]
    # hedge date j can hold month m while it is strictly before the month's first decision date
    open_ = tau_h[:, None] < inp.times[first][None, :] - 1e-12

    def simulate(n: int, rng: np.random.Generator):
        x_all = model.simulate_x(times_all, n, rng)
        paths = simulate_policy(policy, inp, model, x_all[idx_k])
        # inventory at each hedge date: initial before delivery, start-of-step inventory after
        k_of_h = np.searchsorted(inp.times, tau_h - 1e-12)
        inv = np.where(
            (tau_h < inp.times[0] - 1e-12)[:, None],
            contract.initial_inventory,
            paths.inventory[np.minimum(k_of_h, len(inp.times) - 1)],
        )
        fwd = np.empty((len(tau_h), len(months), n))
        for j, (ih, t) in enumerate(zip(idx_h, tau_h, strict=True)):
            for i, (s, w) in enumerate(zip(steps, weights, strict=True)):
                if open_[j, i] or (j > 0 and open_[j - 1, i]):
                    f = model.forward(
                        x_all[ih][:, None], t, inp.times[s][None, :], inp.prices[s][None, :]
                    )
                    fwd[j, i] = f @ w
                else:
                    fwd[j, i] = np.nan
        loss = contract.fuel_loss
        net = np.where(paths.moves > 0, -(1.0 + loss) * paths.moves, -paths.moves)
        exposure = np.stack(
            [(inp.discount[s][:, None] * net[s] * paths.spot[s]).sum(axis=0) for s in steps]
        )  # (months, n): sum_k D_k net_k S_k
        z = np.stack(
            [
                x_all[ih] / np.sqrt(v) if (v := float(model.variance(t))) > 0 else np.zeros(n)
                for ih, t in zip(idx_h, tau_h, strict=True)
            ]
        )
        return paths, fwd, exposure, z, inv / contract.working_volume

    _paths_f, fwd_f, expo_f, z_f, u_f = simulate(n_paths_fit, rng_fit)
    paths_t, fwd_t, _expo_t, z_t, u_t = simulate(n_paths_test, rng_test)

    n_h, n_m = open_.shape
    position = np.zeros((n_h, n_m, n_paths_test))
    initial = np.zeros(n_m)
    for j in range(n_h):
        phi_f = _tensor_basis(z_f[j], u_f[j], degree_z, degree_v)
        phi_t = _tensor_basis(z_t[j], u_t[j], degree_z, degree_v)
        for i in np.flatnonzero(open_[j]):
            target = expo_f[i] / fwd_f[j, i] / d_m[i]
            beta = np.linalg.lstsq(phi_f, target, rcond=None)[0]
            delta = phi_t @ beta
            position[j, i] = -delta
            if j == 0:
                initial[i] = float(np.mean(phi_f @ beta))

    hedge_pnl = np.zeros(n_paths_test)
    costs = np.zeros(n_paths_test)
    n_trades = 0
    for j in range(n_h):
        prev = position[j - 1] if j > 0 else np.zeros((n_m, n_paths_test))
        trade = np.where(open_[j][:, None], position[j] - prev, 0.0)
        n_trades += int(open_[j].sum())
        costs += 0.5 * bid_ask * np.abs(trade).sum(axis=0) * disc_h[j]
        if j + 1 < n_h:
            for i in np.flatnonzero(open_[j]):
                hedge_pnl += position[j, i] * d_m[i] * (fwd_t[j + 1, i] - fwd_t[j, i])
    unhedged = paths_t.total
    return HedgeResult(
        unhedged=unhedged,
        hedged=unhedged + hedge_pnl - costs,
        hedge_pnl=hedge_pnl,
        costs=costs,
        hedge_dates=hedge_dates,
        initial_delta=pd.Series(initial, index=months, name="initial_delta_mwh"),
        n_trades=n_trades,
    )
