# Data

`raw/` and `processed/` are gitignored: no market data is committed to this repository.
Every dataset used in the analysis is listed below with its source, terms and how to obtain it.
Anything not listed here is **synthetic** and labelled as such in code and figures.

| Dataset | Module | Source | Access | Terms |
|---|---|---|---|---|
| TTF front-month history | `gasstorage.data.ttf` | Yahoo Finance, ticker `TTF=F` (via `yfinance`) | automatic, cached to `raw/ttf_front_month.csv` | Yahoo Finance terms of service: personal use, not redistributed (hence not committed) |
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
