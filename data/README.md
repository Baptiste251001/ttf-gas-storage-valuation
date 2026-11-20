# Data

`raw/` and `processed/` are gitignored: no market data is committed to this repository.
Every dataset used in the analysis is listed below with its source, terms and how to obtain it.
Anything not listed here is **synthetic** and labelled as such in code and figures.

| Dataset | Module | Source | Access | Terms |
|---|---|---|---|---|
| TTF front-month history | `gasstorage.data.ttf` | Yahoo Finance, ticker `TTF=F` (via `yfinance`) | automatic, cached to `raw/ttf_front_month.csv` | Yahoo Finance terms of service: personal use, not redistributed (hence not committed) |
| TTF individual monthly futures (Dec-26, Jan-27, Feb-27), daily closes since 2023 | `gasstorage.data.ttf` (`get_contract_history`) | Yahoo Finance, NYMEX symbols `TTF<month code><yy>.NYM` | automatic, cached to `raw/ttf_contracts.csv` | Yahoo Finance terms of service; not committed |
| TTF forward curve (settlements of 2026-11-17) | `gasstorage.data.forward_curve` | EEX TTF Natural Gas Futures settlement prices | manual copy, `raw/eex_ttf_futures_settlements_2026-11-17.csv` | exchange data terms; not committed |
| Aggregated storage levels NL / DE / FR | `gasstorage.data.agsi` | GIE AGSI+ API, https://agsi.gie.eu | free API key in env var `AGSI_API_KEY`, cached to `raw/agsi_<CC>.csv` | free reuse with mandatory attribution "Source: GIE AGSI+" (GIE API manual v007) |
| Synthetic seasonal forward curve | `gasstorage.data.forward_curve` | generated from `market.forward_curve.synthetic` in `configs/*.yaml` | none | **SYNTHETIC**, not market data |

## 1. TTF front-month history (automatic)

```python
from gasstorage.data.ttf import get_ttf_history

ttf = get_ttf_history()  # downloads once, then reads data/raw/ttf_front_month.csv
ttf = get_ttf_history(refresh=True)  # force a new download
```

`TTF=F` is Yahoo's continuous front-month series (EUR/MWh, history from 2017-10-23). Successive
contracts are stitched without roll adjustment. If Yahoo is unreachable, provide any daily series as
`data/raw/ttf_front_month.csv` with columns:

```
date,price_eur_mwh
2024-01-02,31.95
```

## 2. Forward curve (manual)

There is no free, complete download of the monthly TTF curve; settlements are copied from the EEX
market data page (https://www.eex.com/en/market-data/natural-gas/futures, TTF, last 45 days shown).

**Curve used in this project:** EEX TTF Natural Gas Futures settlement prices of **2026-11-17**, all
listed contracts (months Jan-27 to Nov-28, quarters, seasons, calendars), stored unmodified as
`data/raw/eex_ttf_futures_settlements_2026-11-17.csv`:

```
contract,volume_mwh,open_interest_lots,settlement_eur_mwh
Apr-27,7920,3040,71.221
Q2-27,960960,7529,67.702
...
```

Month contracts build the curve. Quarter / season / calendar strips are checked against the
day-weighted average of their months (`strip_consistency`): on 2026-11-17 all covered strips agree
within 0.0014 EUR/MWh. `configs/storage_default.yaml` points to this file and sets
`valuation_date: 2026-11-17`.

To use another trading day, save the EEX (or ICE Endex) table in the same format, all prices from the
**same session**, and update `csv_path` and `valuation_date`. A plain monthly CSV
(`delivery_month,price`, see `data/templates/ttf_forward_curve_template.csv`) is also accepted.
The loader rejects gaps, duplicates and missing prices.

**Without market data** (fresh clone): use `configs/storage_synthetic.yaml`, identical contract on the
SYNTHETIC curve.

## 3. GIE AGSI+ storage levels

1. Register (free) at https://agsi.gie.eu/account and copy the API key from the account page.
2. Set it in the environment, never in a file tracked by git:
   * PowerShell: `$env:AGSI_API_KEY = "<key>"` (or set it as a user environment variable)
   * bash: `export AGSI_API_KEY=<key>`
3. `agsi.load_storage_levels(["NL", "DE", "FR"], start, end)` downloads and caches one CSV per
   country in `data/raw/`. Without a key, cached CSVs still work; otherwise an `AGSIKeyMissingError`
   explains what to do.

Units: storage in TWh, flows in GWh/d, `full_pct` in % of working gas volume. The API is limited to
60 calls per minute; the loader pauses between pages.

## 4. Individual TTF monthly futures (automatic)

Used for the futures volatility term structure (notebook 06). Yahoo serves only the contracts listed at download
time, with their full history: on 2026-11-19 `TTFZ26.NYM`, `TTFF27.NYM` and `TTFG27.NYM` (from 2023-01-03). Their
2026-11-17 closes match the EEX settlements within 0.05 EUR/MWh. Symbol convention: `TTF` + futures month code
(F G H J K M N Q U V X Z) + two-digit year + `.NYM`, see `gasstorage.data.ttf.contract_ticker`.

## 5. EEX Market Data Hub (local scraper, accumulated history)

Daily settlements of all TTF futures (monthly to yearly, physical and financial), the day-ahead spot (`TTFDA`)
and the EEX gas indices (EGSI, EGIX, NDI, ...) are collected with the third-party scraper
https://github.com/dieguisgs/eex-scraper, installed **next to** this repository (`../eex-scraper`, not vendored:
the scraper has no licence). It calls the public JSON API behind the EEX Market Data Hub, which only serves a rolling
window (about 33 sessions, i.e. 45 calendar days, for futures and spot; one year for indices; requesting more, e.g. from 2015, returns the same), so the history builds up by running
it again before the window rolls over (at least every 6 weeks). No scheduled task is installed; run it by hand:

```bash
cd ../eex-scraper && .venv/Scripts/python run.py run      # Windows; .venv/bin/python on Linux/macOS
```

Scraper settings used (its `config.toml`): output `../ttf-gas-storage-valuation/data/raw/eex_hub`, commodity
`NATGAS`, area `TTF`, futures / spot / indices without the 5 MW duplicates, 0.5 request per second, and the browser
fallback **disabled** (if EEX blocks the API, collection stops; no workaround). Data stays in `data/raw/` and is not
redistributed; check the EEX website terms of use for your own use.

Reading it: `gasstorage.data.eex_hub.futures_panel()` (trade date x delivery month), `curve_on(root, date)` (a
`ForwardCurve`), `series(root, "TTFDA")`. A configuration can use it directly with
`market.forward_curve.source: eex_hub`, `csv_path: data/raw/eex_hub`, the trade date being `market.valuation_date`.
First collection (2026-11-20): 33 sessions of monthly physical futures, 2026-10-06 to 2026-11-19, 84 delivery months
(Dec-26 to Oct-33); the 2026-11-17 curve matches the hand-typed EEX table exactly.
