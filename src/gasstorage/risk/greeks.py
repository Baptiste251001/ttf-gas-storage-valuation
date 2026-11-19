r"""Greeks of the LSMC storage value by bump-and-revalue with common random numbers (CRN).

Every bumped valuation re-runs the full LSMC (regressions re-estimated, policy re-optimised) with
the **same seeds**, hence the same standard normal draws: the model state is $X = \sigma \times$
(fixed function of the draws), so bumps of $F(0,\cdot)$, $\sigma$ or $\kappa$ move every path
smoothly and most of the Monte Carlo noise cancels in the differences.

With $V^{\pm}$ the bumped values and $V^{(n)}$ the discounted P&L of valuation path $n$:

* delta per delivery month $m$ (bump $h_m = \epsilon F_m$ of every price in month $m$):
  $\Delta_m = (V^+ - V^-)/(2h_m)$, in MWh (EUR per EUR/MWh);
* parallel delta and aggregated gamma (absolute shift $h$ of the whole curve):
  $\Delta = (V^+ - V^-)/(2h)$, $\Gamma = (V^+ - 2V + V^-)/h^2$;
* vega $\partial V/\partial\sigma$ and sensitivity to mean reversion
  $\partial V/\partial\kappa$ by central differences.

**Bump size.** For a deterministic curve ($\sigma = 0$) the value is piecewise linear in the prices,
with kinks where the optimal schedule changes; a central difference across a kink returns an average
of the one-sided slopes. With $\sigma > 0$ the expectation over paths smooths these kinks. The
default relative bump of 1 % balances this bias against Monte Carlo noise.

Standard errors use the per-path differences, e.g.
$\operatorname{SE}(\Delta_m) = \operatorname{sd}\big((V^{+,(n)} - V^{-,(n)})/(2h_m)\big)/\sqrt N$.
They measure valuation-path noise for fixed regression paths; stability across seeds is tested
separately.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gasstorage.models.one_factor import OneFactorModel
from gasstorage.storage.contract import ValuationInputs
from gasstorage.storage.lsmc import lsmc_value


def bump_month(inp: ValuationInputs, month: pd.Period, rel: float) -> ValuationInputs:
    """Multiply $F(0,t_k)$ by $(1 + \\epsilon)$ for the decision dates of ``month``."""
    mask = inp.dates.to_period("M") == month
    prices = np.where(mask, inp.prices * (1.0 + rel), inp.prices)
    return dataclasses.replace(inp, prices=prices)


def bump_parallel(inp: ValuationInputs, shift: float) -> ValuationInputs:
    """Add ``shift`` EUR/MWh to every $F(0,t_k)$."""
    return dataclasses.replace(inp, prices=inp.prices + shift)


@dataclass(frozen=True)
class Greek:
    value: float
    se: float


@dataclass(frozen=True)
class GreeksResult:
    """Base value and sensitivities. ``deltas`` is indexed by delivery month."""

    base_value: float
    base_se: float
    deltas: pd.DataFrame
    parallel_delta: Greek
    gamma: Greek
    vega: Greek
    kappa_sensitivity: Greek


def _central(up: np.ndarray, down: np.ndarray, width: float) -> Greek:
    d = (up - down) / width
    return Greek(float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size)))


def lsmc_greeks(
    inp: ValuationInputs,
    model: OneFactorModel,
    n_paths: int,
    seeds: tuple[int, int],
    n_levels: int = 101,
    basis: str = "polynomial",
    degree: int = 3,
    rel_bump: float = 0.01,
    parallel_shift: float = 0.5,
    sigma_bump: float = 0.05,
    kappa_bump: float = 0.5,
    months: Sequence[pd.Period] | None = None,
) -> GreeksResult:
    """Bump-and-revalue greeks with CRN (same regression and valuation seeds for every run)."""

    def run(inputs: ValuationInputs, mdl: OneFactorModel) -> np.ndarray:
        return lsmc_value(
            inputs,
            mdl,
            n_paths,
            n_paths,
            np.random.default_rng(seeds[0]),
            np.random.default_rng(seeds[1]),
            n_levels=n_levels,
            basis=basis,
            degree=degree,
        ).path_values

    base = run(inp, model)
    all_months = pd.PeriodIndex(inp.dates.to_period("M").unique())
    months = all_months if months is None else pd.PeriodIndex(list(months), freq="M")

    rows = {}
    for m in months:
        f_m = float(np.mean(inp.prices[inp.dates.to_period("M") == m]))
        h = rel_bump * f_m
        g = _central(
            run(bump_month(inp, m, rel_bump), model),
            run(bump_month(inp, m, -rel_bump), model),
            2 * h,
        )
        rows[m] = {"forward": f_m, "delta_mwh": g.value, "se": g.se}
    deltas = pd.DataFrame(
        list(rows.values()),
        index=pd.PeriodIndex(list(rows), freq="M", name="month"),
        columns=["forward", "delta_mwh", "se"],
    )

    up = run(bump_parallel(inp, parallel_shift), model)
    down = run(bump_parallel(inp, -parallel_shift), model)
    gamma_paths = (up - 2 * base + down) / parallel_shift**2
    gamma = Greek(float(gamma_paths.mean()), float(gamma_paths.std(ddof=1) / np.sqrt(n_paths)))

    s, k = model.sigma, model.kappa
    vega = _central(
        run(inp, OneFactorModel(k, s + sigma_bump)),
        run(inp, OneFactorModel(k, max(s - sigma_bump, 0.0))),
        s + sigma_bump - max(s - sigma_bump, 0.0),
    )
    kappa_sens = _central(
        run(inp, OneFactorModel(k + kappa_bump, s)),
        run(inp, OneFactorModel(k - kappa_bump, s)),
        2 * kappa_bump,
    )
    return GreeksResult(
        base_value=float(base.mean()),
        base_se=float(base.std(ddof=1) / np.sqrt(n_paths)),
        deltas=deltas,
        parallel_delta=_central(up, down, 2 * parallel_shift),
        gamma=gamma,
        vega=vega,
        kappa_sensitivity=kappa_sens,
    )
