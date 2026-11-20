"""Valuation snapshots used to compare runs, e.g. the same contract on two curve dates.

A snapshot gathers, for one configuration (one forward curve and valuation date):

* a summary of the curve over the storage year: day-weighted averages of summer (April to
  September) and winter (October to March) prices, their spread and whether the curve is strictly
  decreasing (then no calendar spread is profitable and the intrinsic value is zero);
* the intrinsic value (weekly MILP and the DP on the LSMC grid);
* for each price model, the LSMC value (out-of-sample lower bound) with its standard error, and
  optionally, for one-factor models, the lower and dual upper bounds of the grid-restricted problem.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from gasstorage.config import Config
from gasstorage.models.base import PriceModel
from gasstorage.storage.contract import ValuationInputs, build_inputs
from gasstorage.storage.dp import solve_dp
from gasstorage.storage.dual import dual_upper_bound
from gasstorage.storage.intrinsic import solve_lp
from gasstorage.storage.lsmc import lsmc_value

_SUMMER = (4, 5, 6, 7, 8, 9)


def curve_summary(inp: ValuationInputs) -> dict[str, Any]:
    """Day-weighted summer / winter averages of $F(0,t_k)$ over the decision dates."""
    summer = np.isin(inp.dates.month, _SUMMER)

    def avg(mask: np.ndarray) -> float:
        return float(np.sum(inp.prices[mask] * inp.days[mask]) / np.sum(inp.days[mask]))

    s, w = avg(summer), avg(~summer)
    monthly = inp.curve.to_frame().set_index("delivery_month")["price"]
    in_year = monthly.loc[str(inp.dates[0].to_period("M")) : str(inp.dates[-1].to_period("M"))]
    return {
        "curve_source": inp.curve.source,
        "summer_avg_eur_mwh": s,
        "winter_avg_eur_mwh": w,
        "winter_minus_summer_eur_mwh": w - s,
        "first_month_eur_mwh": float(in_year.iloc[0]),
        "last_month_eur_mwh": float(in_year.iloc[-1]),
        "strictly_decreasing": bool(np.all(np.diff(in_year.to_numpy()) < 0)),
        "monthly_eur_mwh": {str(k): float(v) for k, v in in_year.items()},
    }


def valuation_snapshot(
    cfg: Config,
    models: Mapping[str, PriceModel],
    n_paths: int,
    n_levels: int = 101,
    dual_for: Iterable[str] = (),
) -> dict[str, Any]:
    """Curve summary, intrinsic value and LSMC (and dual bounds) per model, weekly decisions."""
    inp = build_inputs(cfg, "weekly")
    mc = cfg.monte_carlo
    k = inp.contract

    def rngs() -> tuple[np.random.Generator, np.random.Generator]:
        return np.random.default_rng(mc.seed_regression), np.random.default_rng(mc.seed_valuation)

    out: dict[str, Any] = {
        "valuation_date": str(cfg.market.valuation_date),
        "curve": curve_summary(inp),
        "intrinsic_milp_eur": solve_lp(k, inp.prices, inp.days, inp.discount, ratchets=True).value,
        "intrinsic_same_grid_eur": solve_dp(k, inp.prices, inp.days, inp.discount, n_levels).value,
        "settings": {"n_paths": n_paths, "n_levels": n_levels, "step": "weekly"},
        "models": {},
    }
    dual_for = set(dual_for)
    for name, model in models.items():
        res = lsmc_value(inp, model, n_paths, n_paths, *rngs(), n_levels=n_levels)
        row: dict[str, Any] = {"lsmc_eur": res.value, "lsmc_se_eur": res.se}
        if name in dual_for:
            lb = lsmc_value(inp, model, n_paths, n_paths, *rngs(), n_levels=n_levels, moves="grid")
            ub = dual_upper_bound(inp, model, lb.policy, n_paths, np.random.default_rng(24680))
            row |= {
                "grid_lower_eur": lb.value,
                "grid_lower_se_eur": lb.se,
                "dual_upper_eur": ub.value,
                "dual_se_eur": ub.se,
            }
        out["models"][name] = row
    return out
