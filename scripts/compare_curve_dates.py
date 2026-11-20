"""Revalue the storage on EEX hub curves of several trade dates and compare the runs.

Usage:
    python scripts/compare_curve_dates.py --dates 2026-11-17 2026-11-19 --paths 10000

Same contract and parameters for every date; only the forward curve (and the valuation date)
change. Writes one JSON per date and a Markdown comparison to ``reports/runs/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from gasstorage.config import load_config
from gasstorage.data import eex_hub
from gasstorage.data.ttf import get_contract_history, get_ttf_history
from gasstorage.models.one_factor import OneFactorModel
from gasstorage.models.two_factor import TwoFactorModel
from gasstorage.models.vol_term_structure import (
    fit_one_factor_kappa,
    fit_two_factor,
    vol_ratio_table,
)
from gasstorage.report import valuation_snapshot


def _models(cfg) -> dict:
    contracts = get_contract_history([])  # cached in data/raw/ttf_contracts.csv
    starts = {c: pd.Period(c, "M").start_time for c in contracts.columns}
    table = vol_ratio_table(contracts, starts, get_ttf_history())
    k_q = fit_one_factor_kappa(table).kappa
    two = fit_two_factor(table, front_vol=cfg.model.sigma)
    return {
        "1F historical (P)": OneFactorModel(cfg.model.kappa, cfg.model.sigma),
        "1F risk-neutral (Q)": OneFactorModel(k_q, cfg.model.sigma),
        "2F Schwartz-Smith (Q)": TwoFactorModel(two.kappa, two.sigma_chi, two.sigma_xi, two.rho),
    }


def _config_for(base, hub: Path, day: str):
    fc = dataclasses.replace(base.market.forward_curve, source="eex_hub", csv_path=hub)
    market = dataclasses.replace(
        base.market, valuation_date=pd.Timestamp(day).date(), forward_curve=fc
    )
    return dataclasses.replace(base, market=market)


def _meur(x: float) -> str:
    return f"{x / 1e6:.3f}"


def _markdown(snaps: dict[str, dict]) -> str:
    dates = list(snaps)
    lines = [
        f"# Storage valuation on EEX hub curves: {' vs '.join(dates)}",
        "",
        "Same contract (storage year 2027/28) and parameters; only the forward curve and the "
        "valuation date change. Weekly decisions, "
        f"{snaps[dates[0]]['settings']['n_levels']} volume levels, "
        f"{snaps[dates[0]]['settings']['n_paths']:,} + {snaps[dates[0]]['settings']['n_paths']:,} "
        "LSMC paths, fixed seeds. MEUR unless stated; SE in brackets.",
        "",
        "## Curve over the storage year (EUR/MWh)",
        "",
        "| | " + " | ".join(dates) + " | change |",
        "|---|" + "---:|" * (len(dates) + 1),
    ]
    c = [snaps[d]["curve"] for d in dates]
    for key, label in [
        ("first_month_eur_mwh", "Apr-27"),
        ("last_month_eur_mwh", "Mar-28"),
        ("summer_avg_eur_mwh", "summer average (Apr-Sep)"),
        ("winter_avg_eur_mwh", "winter average (Oct-Mar)"),
        ("winter_minus_summer_eur_mwh", "winter - summer spread"),
    ]:
        vals = [x[key] for x in c]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{v:.3f}" for v in vals)
            + f" | {vals[-1] - vals[0]:+.3f} |"
        )
    lines.append(
        "| strictly decreasing | " + " | ".join(str(x["strictly_decreasing"]) for x in c) + " | |"
    )
    lines += ["", "## Values (MEUR)", "", "| | " + " | ".join(dates) + " | change |"]
    lines.append("|---|" + "---:|" * (len(dates) + 1))
    v = [snaps[d]["intrinsic_milp_eur"] for d in dates]
    lines.append(
        "| intrinsic (MILP) | " + " | ".join(_meur(x) for x in v) + f" | {_meur(v[-1] - v[0])} |"
    )
    for name in snaps[dates[0]]["models"]:
        m = [snaps[d]["models"][name] for d in dates]
        cells = [f"{_meur(x['lsmc_eur'])} ({_meur(x['lsmc_se_eur'])})" for x in m]
        diff = m[-1]["lsmc_eur"] - m[0]["lsmc_eur"]
        lines.append(f"| LSMC, {name} | " + " | ".join(cells) + f" | {_meur(diff)} |")
        if "dual_upper_eur" in m[0]:
            cells = [f"{_meur(x['dual_upper_eur'])} ({_meur(x['dual_se_eur'])})" for x in m]
            diff = m[-1]["dual_upper_eur"] - m[0]["dual_upper_eur"]
            lines.append(
                f"| dual upper bound, {name} | " + " | ".join(cells) + f" | {_meur(diff)} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/storage_default.yaml")
    p.add_argument("--hub", type=Path, default=eex_hub.DEFAULT_ROOT)
    p.add_argument("--dates", nargs="+", help="trade dates (default: 2026-11-17 and latest)")
    p.add_argument("--paths", type=int, default=10_000)
    p.add_argument("--levels", type=int, default=101)
    p.add_argument("--out", type=Path, default=Path("reports/runs"))
    args = p.parse_args(argv)

    base = load_config(args.config)
    latest = eex_hub.futures_panel(args.hub).index.max().date()
    dates = args.dates or ["2026-11-17", str(latest)]
    models = _models(base)
    snaps = {}
    for day in dates:
        cfg = _config_for(base, args.hub, day)
        snaps[day] = valuation_snapshot(
            cfg, models, args.paths, args.levels, dual_for=["1F risk-neutral (Q)"]
        )
        print(f"{day}: done")
    args.out.mkdir(parents=True, exist_ok=True)
    for day, snap in snaps.items():
        (args.out / f"run_{day}.json").write_text(json.dumps(snap, indent=2, default=float))
    md = _markdown(snaps)
    (args.out / f"compare_{'_vs_'.join(dates)}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
