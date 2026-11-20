# CLAUDE.md

Portfolio project: valuation of a natural gas storage contract on the TTF hub as a real option.
Intrinsic value (LP/DP on the forward curve) + extrinsic value (LSMC, Boogert & de Jong 2008),
greeks and delta hedging. Target audience: energy trading desks (quant / analyst roles).

## Working rules

1. **Language**: code, docstrings, comments, README, notebooks in English. Talk to the user in French.
2. **Phases**: work phase by phase (see below). End of each phase: run ruff + pytest, make a
   conventional commit (`feat(lsmc): ...`, `chore: ...`), then STOP and summarise (what was done,
   numerical results, open choices). Never start the next phase without explicit go-ahead.
3. **No invented data**: never write prices, forward curves or "real" parameters from memory.
   Data comes from a downloaded, documented source (`data/README.md`) or is explicitly labelled
   **synthetic**. If a source is not reachable, build a CSV loader and tell the user what to download.
4. **Rigour**: every implemented formula is written in LaTeX in the docstring or notebook.
   Monte Carlo results are always reported with their standard error and a fixed seed.
   LSMC with independent regression/valuation paths is a **lower bound**: say so.
5. **Quality**: Python 3.11, type hints, `ruff check` + `ruff format`, `pytest`. Vectorised numpy,
   no Python loop over paths. No business logic in notebooks: they import `gasstorage`.
6. **TDD**: write the failing test first, watch it fail, then implement.
7. Contract parameters are **illustrative**, not a real site.
8. **Commits**: conventional messages, **no `Co-Authored-By: Claude` trailer** or any Claude
   attribution (history was rewritten to remove them). Do not care about GitHub CI status.

## Environment

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e ".[dev,data]"
.venv/Scripts/python -m pytest          # Windows; .venv/bin/python on Linux/macOS
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
```

Mark long Monte Carlo tests with `@pytest.mark.slow`; CI runs `pytest -m "not slow"`.
Secrets: `AGSI_API_KEY` env var only, never committed.

## Architecture

```
configs/storage_default.yaml   contract / market / model / monte_carlo parameters
src/gasstorage/
  config.py                    frozen, validated dataclasses; load_config(path) -> Config
  data/                        ttf.py (history), agsi.py (GIE AGSI+), forward_curve.py (CSV + synthetic)
  models/                      seasonality, one_factor (OU on log-price), two_factor (Schwartz-Smith), calibration
  storage/                     contract, intrinsic (cvxpy LP/MILP), dp (volume-grid DP), rolling_intrinsic,
                               basis (poly / Laguerre), lsmc
  risk/                        greeks (bump & revalue, CRN), delta_hedge
  plotting.py                  shared matplotlib style
notebooks/01..05               exploration, calibration, intrinsic, LSMC, greeks & hedging
scripts/run_valuation.py       CLI
reports/figures/               committed PNGs used by README
tests/                         one test file per area
```

Config conventions: volumes in MWh, prices/costs in EUR/MWh, rates in MWh/day, time in years
for model parameters. Ratchets are piecewise-constant tiers `{lower, upper, factor}` on the fill
level (inventory / working volume), contiguous over [0, 1], factor in (0, 1].

## Phases

1. Scaffold (done) 2. Data (done) 3. One-factor price model (done) 4. Intrinsic (LP, DP, rolling intrinsic) (done)
5. LSMC (core) (done)
6. Risk + results-first README (done)
7. Extensions (done, Streamlit excluded by the user): Schwartz-Smith,
Andersen-Broadie dual upper bound, AGSI comparison, Streamlit app.

## Key model facts

- One factor: $\ln S_t = \ln F(0,t) + X_t - \tfrac12 \mathrm{Var}(X_t)$, $dX_t = -\kappa X_t dt + \sigma dW_t$,
  so $E[S_t] = F(0,t)$. Exact OU simulation (no Euler).
- $F(t,T) = F(0,T)\exp(e^{-\kappa(T-t)}X_t + \tfrac12 V(t,T) - \tfrac12 V(0,T))$,
  $V(t,T) = \tfrac{\sigma^2}{2\kappa}(1-e^{-2\kappa(T-t)})$. Derive and verify in notebook 02.
- Calibration under the historical measure; discuss vs risk-neutral and calibration window.

## Decisions taken with the user

- Contract = storage year 2027/28 (2027-04-01 to 2028-03-31). `valuation_date` = settlement date of
  the hand-typed forward curve (`data/raw/ttf_forward_curve.csv`); synthetic curve until it exists.
- Calibration: **reference window = post-crisis, 2023-04 to date**, log-price minus linear trend +
  2 Fourier harmonics, exact MLE, dt = 1/252: kappa = 4.784 (se 1.585), sigma = 0.694 (se 0.016),
  half-life 53 days. Sensitivity on the full sample with the 2020 collapse, the 2021-2023 crisis and
  the 2026 rally flagged. kappa is weakly identified (DF t-stat -3.0): report valuation sensitivity
  to kappa in Phase 6; without the trend kappa = 2.7.
- **Two curve cases, both kept and explained everywhere (notebooks 03-05, README):**
  1. *Market case* (`storage_default.yaml`): EEX TTF settlements of 2026-11-17. Storage year 2027/28
     is strictly backwardated, so intrinsic value = 0 and the whole value is extrinsic. This is the
     real market answer and the headline result.
  2. *Contango case* (`storage_synthetic.yaml`): SYNTHETIC seasonal curve with a summer-winter spread.
     It shows what the market case cannot: positive intrinsic value, the LP = DP cross-check on a
     non-trivial schedule, the intrinsic / extrinsic split, and rolling intrinsic vs LSMC.
- AGSI+ key: only via the `AGSI_API_KEY` env var; data cached in `data/raw/agsi_<CC>.csv`.
- Notebooks are executed with `jupyter nbconvert --to notebook --execute --inplace` and committed
  with outputs; figures go to `reports/figures/`.
- Storage engine conventions (Phase 4): ratchets act on the start-of-step fill level; volumes of
  step k trade at the price of its first day; hard terminal constraint in the DP = penalty 1e4 EUR/MWh.
  The grid DP (`storage/dp.py`, `best_moves`) is the shared engine for intrinsic, rolling intrinsic
  and LSMC. With tier ratchets the DP converges to the MILP from below, error O(grid step).
- HiGHS prints a banner once per process; it cannot be silenced through cvxpy (`output_flag` ignored).
- LSMC (Phase 5): Longstaff-Schwartz style (realised values propagated backwards), one regression per
  grid level on polynomials of z = X / sqrt(V(0,t)) (degree 3); independent regression / valuation
  seeds (enforced). Out-of-sample value = lower bound. Backward pass uses `GridMoves` (path-independent
  action set, bit-identical to `best_moves`): 10k + 10k weekly paths in ~25 s. Grid size is the
  dominant numerical error (O(grid step), ~2 % at 101 levels); state it with results.
- Headline (weekly, G=101, 10k+10k): market 4.90 MEUR (SE 0.06), all extrinsic; contango 7.98 MEUR,
  intrinsic 6.43 on the same grid. Notebook 04 takes ~12 min to execute (convergence studies).
- Risk (Phase 6): greeks = full LSMC re-runs with CRN (same seeds); bump 1 % per month, 0.5 EUR parallel,
  +-0.05 sigma, +-0.5 kappa. With sigma = 0 bumps must not cross a schedule change (kinks).
  Delta hedge ratio = envelope theorem, regressed on (X, inventory) over fit paths independent of the
  test paths; monthly forwards stop at the first decision date of their month.
- Tests needing the EEX CSV call `tests.conftest.skip_without_market_data()` so CI (no data) skips them.
- CLI: `gasstorage.cli.main`, wrapped by `scripts/run_valuation.py`. README numbers come from
  notebooks 04 (valuation, 10k+10k) and 05 (greeks / hedge, 5k+5k); keep them in sync if rerun.
- GitHub: https://github.com/Baptiste251001/ttf-gas-storage-valuation (public). Companion repo:
  https://github.com/Baptiste251001/deep-hedging-transaction-costs. Push only when the user asks.
- Phase 7 results: futures vol term structure (3 Yahoo NYMEX contracts) gives kappa_Q = 0.33 +- 0.02
  vs kappa_P = 4.78; Schwartz-Smith fit kappa 0.46, sigma_chi 0.74, sigma_xi 0.27, rho -0.31 (weak).
  Under Q the market-case value collapses: LSMC -0.13, RI 0.04, dual upper bound 0.16 MEUR (vs 4.90 under P).
  Dual gap under P: 1.5-2.7 %. Engines take any model with the `models.base.PriceModel` interface; the
  2F model with sigma_xi = 0 reproduces 1F bit for bit (same op order, vander powers). Dual bound only
  for 1F + polynomial basis + grid-restricted moves (`moves="grid"`); grid step must be <= smallest
  ratcheted move (use >= 101 levels).
- README headline (decided with the user): risk-neutral range 0-0.3 MEUR is the main result; the
  4.90 MEUR value under historical front-month dynamics is presented as a scenario.
- EEX hub data (2026-11-20): third-party scraper in ../eex-scraper (reviewed, no licence, not vendored),
  config: NATGAS/TTF, F+S+I without 5MW, 0.5 rps, browser fallback disabled (never bypass a block).
  NO scheduled task (user refused); run by hand at least every 6 weeks. Reader: `data/eex_hub.py`;
  curve source `eex_hub` (csv_path = hub root, trade date = valuation_date).
- Spot vs front (EGSI TTF Day vs Yahoo TTF=F, 2025-11-20..2026-11-18, 250 days): log basis std 1.4 %,
  OU half-life 0.7 days (DF -12). Fast spot factor is real but small: it does not explain the 4.9 MEUR
  historical-dynamics value, which comes from slow front-month level reversion.
