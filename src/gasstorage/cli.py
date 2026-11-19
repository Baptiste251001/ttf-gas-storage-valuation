"""Command-line valuation.

Usage: ``python scripts/run_valuation.py --config configs/storage_default.yaml``.

Prints the intrinsic, LSMC (out-of-sample lower bound, with standard error) and extrinsic values,
optionally the rolling intrinsic value, greeks and delta-hedge statistics, and can write
everything to JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gasstorage.config import load_config
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.risk.delta_hedge import delta_hedge, pnl_statistics
from gasstorage.risk.greeks import lsmc_greeks
from gasstorage.storage.contract import build_inputs
from gasstorage.storage.dp import solve_dp
from gasstorage.storage.intrinsic import solve_lp
from gasstorage.storage.lsmc import lsmc_value
from gasstorage.storage.rolling_intrinsic import rolling_intrinsic


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Value a gas storage contract on the TTF (LSMC).")
    p.add_argument("--config", required=True, help="YAML configuration file")
    p.add_argument("--step", choices=["daily", "weekly"], help="override monte_carlo.time_step")
    p.add_argument("--paths", type=int, help="override both regression and valuation path counts")
    p.add_argument("--levels", type=int, help="override monte_carlo.volume_grid_size")
    p.add_argument("--rolling-intrinsic", type=int, metavar="N", help="also run rolling intrinsic")
    p.add_argument("--greeks", action="store_true", help="also compute greeks (slow)")
    p.add_argument("--hedge", choices=["monthly", "weekly"], help="also simulate a delta hedge")
    p.add_argument("--bid-ask", type=float, default=0.0, help="hedge bid-ask spread, EUR/MWh")
    p.add_argument("--json", type=Path, help="write results to this JSON file")
    return p


def _meur(x: float) -> str:
    return f"{x / 1e6:10.3f} MEUR"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    mc = cfg.monte_carlo
    step = args.step or mc.time_step
    n_reg = args.paths or mc.n_paths_regression
    n_val = args.paths or mc.n_paths_valuation
    levels = args.levels or mc.volume_grid_size
    inp = build_inputs(cfg, step)
    model = OneFactorModel.from_config(cfg.model)

    t0 = time.perf_counter()
    lsmc = lsmc_value(
        inp, model, n_reg, n_val,
        np.random.default_rng(mc.seed_regression), np.random.default_rng(mc.seed_valuation),
        n_levels=levels, basis=mc.basis, degree=mc.basis_degree,
    )  # fmt: skip
    elapsed = time.perf_counter() - t0
    intrinsic_grid = solve_dp(inp.contract, inp.prices, inp.days, inp.discount, levels).value
    intrinsic_milp = solve_lp(inp.contract, inp.prices, inp.days, inp.discount, ratchets=True).value

    result: dict[str, Any] = {
        "config": str(args.config),
        "curve": {"source": inp.curve.source, "is_synthetic": inp.curve.is_synthetic},
        "settings": {
            "step": step, "volume_levels": levels, "n_paths_regression": n_reg,
            "n_paths_valuation": n_val, "seed_regression": mc.seed_regression,
            "seed_valuation": mc.seed_valuation, "basis": mc.basis, "degree": mc.basis_degree,
            "kappa": model.kappa, "sigma": model.sigma,
        },
        "valuation": {
            "lsmc_value_eur": lsmc.value,
            "lsmc_se_eur": lsmc.se,
            "lsmc_in_sample_eur": lsmc.in_sample_value,
            "intrinsic_same_grid_eur": intrinsic_grid,
            "intrinsic_milp_eur": intrinsic_milp,
            "extrinsic_eur": lsmc.value - intrinsic_grid,
            "run_time_s": elapsed,
        },
    }  # fmt: skip

    print(f"Curve: {inp.curve.source}{'  [SYNTHETIC]' if inp.curve.is_synthetic else ''}")
    print(f"Grid: {step}, {len(inp.prices)} steps, {levels} volume levels; paths {n_reg}+{n_val}")
    print(f"Intrinsic (MILP)        {_meur(intrinsic_milp)}")
    print(f"Intrinsic (same grid)   {_meur(intrinsic_grid)}")
    print(f"LSMC value              {_meur(lsmc.value)}  +/- {lsmc.se / 1e6:.3f} (1 SE)")
    print(f"Extrinsic               {_meur(lsmc.value - intrinsic_grid)}")
    print(f"  (LSMC is an out-of-sample lower bound; run time {elapsed:.1f} s)")

    if args.rolling_intrinsic:
        ri = rolling_intrinsic(
            inp, model, args.rolling_intrinsic, np.random.default_rng(2027), levels
        )
        result["rolling_intrinsic"] = {"value_eur": ri.value, "se_eur": ri.se}
        print(f"Rolling intrinsic       {_meur(ri.value)}  +/- {ri.se / 1e6:.3f}")

    if args.greeks:
        g = lsmc_greeks(
            inp, model, min(n_reg, n_val), (mc.seed_regression, mc.seed_valuation), levels,
            mc.basis, mc.basis_degree,
        )  # fmt: skip
        result["greeks"] = {
            "parallel_delta_mwh": [g.parallel_delta.value, g.parallel_delta.se],
            "gamma_mwh_per_eur": [g.gamma.value, g.gamma.se],
            "vega_eur_per_unit_sigma": [g.vega.value, g.vega.se],
            "dv_dkappa_eur": [g.kappa_sensitivity.value, g.kappa_sensitivity.se],
            "monthly_delta_mwh": {str(m): [r.delta_mwh, r.se] for m, r in g.deltas.iterrows()},
        }
        pdelta, vega = g.parallel_delta, g.vega
        print(f"Parallel delta          {pdelta.value / 1e3:10.1f} GWh  +/- {pdelta.se / 1e3:.1f}")
        print(f"Vega (per 0.01 sigma)   {vega.value / 1e5:10.1f} kEUR +/- {vega.se / 1e5:.1f}")

    if args.hedge:
        h = delta_hedge(
            inp, model, lsmc.policy, n_reg, n_val,
            np.random.default_rng(mc.seed_regression + 1),
            np.random.default_rng(mc.seed_valuation + 1),
            rebalance=args.hedge, bid_ask=args.bid_ask,
        )  # fmt: skip
        stats = pnl_statistics({"unhedged": h.unhedged, "hedged": h.hedged})
        result["hedge"] = {
            "rebalance": args.hedge, "bid_ask": args.bid_ask,
            "statistics_eur": stats.to_dict(orient="index"),
        }  # fmt: skip
        print((stats / 1e6).round(3).to_string())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=float))
    return 0
