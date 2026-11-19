# Data

`raw/` and `processed/` are gitignored: no market data is committed to this repository.
Every dataset used in the analysis is listed below with its source, licence and how to obtain it.
Anything not listed here is **synthetic** and labelled as such in code and figures.

| Dataset | Module | Source | Access | Licence / terms |
|---|---|---|---|---|
| TTF front-month history | `gasstorage.data.ttf` | Yahoo Finance via `yfinance` (to be confirmed in Phase 2), CSV fallback | automatic / manual | to be documented in Phase 2 |
| TTF forward curve (monthly settlements) | `gasstorage.data.forward_curve` | ICE Endex or EEX settlement pages, typed by hand | manual CSV: `delivery_month, price` | to be documented in Phase 2 |
| Aggregated storage levels NL/DE/FR | `gasstorage.data.agsi` | GIE AGSI+ API | API key in env var `AGSI_API_KEY` | to be documented in Phase 2 |
| Synthetic seasonal forward curve | `gasstorage.data.forward_curve` | generated (`configs/*.yaml`, `market.forward_curve.synthetic`) | n/a | **SYNTHETIC** |

Download instructions are completed in Phase 2.
