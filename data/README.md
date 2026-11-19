# Data

`raw/` and `processed/` are gitignored: no market data is committed to this repository.
Every dataset used in the analysis is listed below with its source, terms and how to obtain it.
Anything not listed here is **synthetic** and labelled as such in code and figures.

| Dataset | Module | Source | Access | Terms |
|---|---|---|---|---|
| TTF front-month history | `gasstorage.data.ttf` | Yahoo Finance, ticker `TTF=F` (via `yfinance`) | automatic, cached to `raw/ttf_front_month.csv` | Yahoo Finance terms of service: personal use, not redistributed (hence not committed) |
| TTF forward curve (monthly settlements) | `gasstorage.data.forward_curve` | ICE Endex Dutch TTF Gas Futures or EEX TTF futures settlement prices | manual CSV `raw/ttf_forward_curve.csv` | exchange data terms; typed by hand, not committed |
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

There is no free, complete source of the monthly TTF curve. To value on a real curve:

1. Copy `data/templates/ttf_forward_curve_template.csv` to `data/raw/ttf_forward_curve.csv`.
2. Fill the `price` column (EUR/MWh) with the **settlement prices of the monthly contracts** for every
   delivery month of the storage year, all from the **same trading day**, from the ICE Endex
   "Dutch TTF Natural Gas Futures" settlement page or the EEX TTF futures page.
3. In the config, set `market.forward_curve.source: csv`, `csv_path: data/raw/ttf_forward_curve.csv`,
   `market.valuation_date` to that trading day, and the contract dates to the storage year.

Format: `delivery_month` as `YYYY-MM`, months contiguous, one row each, prices positive. The loader
rejects gaps, duplicates and missing prices. If only quarter / season contracts are quoted for the
far months, enter the quarter or season price for each month it covers and say so in the analysis.

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
