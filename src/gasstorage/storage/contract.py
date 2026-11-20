r"""Storage contract and the deterministic inputs of a valuation run.

Notation (step $k = 0,\dots,K-1$ of length $d_k$ days, inventory $v_k$ at the start of step $k$,
working volume $\bar V$, fill level $\ell_k = v_k / \bar V$):

* inventory dynamics $v_{k+1} = v_k + i_k - w_k$, with $0 \le v_k \le \bar V$;
* rate limits with ratchets evaluated at the start-of-step fill level,
  $i_k \le q^{\text{inj}} d_k \,\phi^{\text{inj}}(\ell_k)$ and
  $w_k \le q^{\text{wdr}} d_k \,\phi^{\text{wdr}}(\ell_k)$,
  where $\phi$ is piecewise constant on half-open tiers $[a_j, b_j)$ (last tier closed);
* cash flow of a volume change $\Delta v = i - w$ traded at price $P$ with discount factor $D$:

  .. math::
     \pi(\Delta v; P, D) = D\Big[-\big(P(1+\lambda) + c^{\text{inj}}\big)\,\Delta v^+
                               + \big(P - c^{\text{wdr}}\big)\,\Delta v^-\Big],

  where $\lambda$ is the fuel loss (gas bought per MWh stored is $1+\lambda$);
* terminal condition $v_K = v^{\text{end}}$, hard, or soft with penalty $p\,|v_K - v^{\text{end}}|$.

Ratchets evaluated at the start of each step are exact for daily steps and an approximation for
weekly steps (the true rate would change inside the week when a tier boundary is crossed).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from gasstorage.config import Config, ContractConfig, RatchetTier
from gasstorage.data.forward_curve import ForwardCurve
from gasstorage.timegrid import decision_grid, year_fraction

_FLAT = (RatchetTier(0.0, 1.0, 1.0),)


def _tier_factor(tiers: tuple[RatchetTier, ...], fill: ArrayLike) -> np.ndarray:
    uppers = np.array([t.upper for t in tiers[:-1]])
    factors = np.array([t.factor for t in tiers])
    idx = np.searchsorted(uppers, np.asarray(fill, dtype=float), side="right")
    return factors[idx]


@dataclass(frozen=True)
class StorageContract:
    """Storage contract terms in MWh, MWh/day and EUR/MWh."""

    working_volume: float
    initial_inventory: float
    terminal_inventory: float
    terminal_penalty: float | None
    max_injection_per_day: float
    max_withdrawal_per_day: float
    injection_ratchets: tuple[RatchetTier, ...]
    withdrawal_ratchets: tuple[RatchetTier, ...]
    injection_cost: float
    withdrawal_cost: float
    fuel_loss: float

    @classmethod
    def from_config(cls, cfg: ContractConfig) -> StorageContract:
        return cls(
            working_volume=cfg.working_volume_mwh,
            initial_inventory=cfg.initial_inventory_mwh,
            terminal_inventory=cfg.terminal_inventory_mwh,
            terminal_penalty=cfg.terminal_penalty_eur_mwh,
            max_injection_per_day=cfg.max_injection_mwh_per_day,
            max_withdrawal_per_day=cfg.max_withdrawal_mwh_per_day,
            injection_ratchets=cfg.injection_ratchets,
            withdrawal_ratchets=cfg.withdrawal_ratchets,
            injection_cost=cfg.injection_cost_eur_mwh,
            withdrawal_cost=cfg.withdrawal_cost_eur_mwh,
            fuel_loss=cfg.fuel_loss,
        )

    @property
    def has_ratchets(self) -> bool:
        return any(t.factor < 1.0 for t in self.injection_ratchets + self.withdrawal_ratchets)

    def without_ratchets(self) -> StorageContract:
        """Same contract with nominal (unratcheted) rates at every fill level."""
        return dataclasses.replace(self, injection_ratchets=_FLAT, withdrawal_ratchets=_FLAT)

    def injection_factor(self, fill: ArrayLike) -> np.ndarray:
        r"""Injection ratchet multiplier $\phi^{\text{inj}}(\ell)$."""
        return _tier_factor(self.injection_ratchets, fill)

    def withdrawal_factor(self, fill: ArrayLike) -> np.ndarray:
        r"""Withdrawal ratchet multiplier $\phi^{\text{wdr}}(\ell)$."""
        return _tier_factor(self.withdrawal_ratchets, fill)

    def max_injection(self, inventory: ArrayLike, days: ArrayLike) -> np.ndarray:
        """Ratcheted injection limit over a step (capacity not included)."""
        fill = np.asarray(inventory, dtype=float) / self.working_volume
        return self.max_injection_per_day * np.asarray(days) * self.injection_factor(fill)

    def max_withdrawal(self, inventory: ArrayLike, days: ArrayLike) -> np.ndarray:
        """Ratcheted withdrawal limit over a step (inventory not included)."""
        fill = np.asarray(inventory, dtype=float) / self.working_volume
        return self.max_withdrawal_per_day * np.asarray(days) * self.withdrawal_factor(fill)

    def cash_flow(self, dv: ArrayLike, price: ArrayLike, discount: ArrayLike) -> np.ndarray:
        r"""Discounted cash flow $\pi(\Delta v; P, D)$ of a volume change (see module docstring)."""
        dv = np.asarray(dv, dtype=float)
        price = np.asarray(price, dtype=float)
        buy = price * (1.0 + self.fuel_loss) + self.injection_cost
        sell = price - self.withdrawal_cost
        return np.asarray(discount) * (-buy * np.maximum(dv, 0.0) + sell * np.maximum(-dv, 0.0))


@dataclass(frozen=True)
class ValuationInputs:
    """Deterministic inputs of a valuation: decision grid, $F(0,t_k)$, model times, discounting."""

    contract: StorageContract
    curve: ForwardCurve
    valuation_date: dt.date
    dates: pd.DatetimeIndex
    days: np.ndarray
    times: np.ndarray
    prices: np.ndarray
    discount: np.ndarray


def build_inputs(cfg: Config, step: str | None = None) -> ValuationInputs:
    r"""Inputs for ``cfg`` on a daily or weekly grid (default: ``cfg.monte_carlo.time_step``).

    $\tau_k$ is ACT/365 from the valuation date and $D_k = e^{-r\tau_k}$.
    """
    c = cfg.contract
    grid = decision_grid(c.start_date, c.end_date, step or cfg.monte_carlo.time_step)
    curve = ForwardCurve.from_config(
        cfg.market.forward_curve, c.start_date, c.end_date, as_of=cfg.market.valuation_date
    )
    times = year_fraction(cfg.market.valuation_date, grid.dates)
    return ValuationInputs(
        contract=StorageContract.from_config(c),
        curve=curve,
        valuation_date=cfg.market.valuation_date,
        dates=grid.dates,
        days=grid.days,
        times=times,
        prices=curve.daily(grid.dates),
        discount=np.exp(-cfg.market.discount_rate * times),
    )
