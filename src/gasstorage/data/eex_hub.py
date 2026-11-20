"""Reader for the local output of the EEX Market Data Hub scraper.

The scraper (https://github.com/dieguisgs/eex-scraper, run separately, not part of this package)
queries the public JSON API behind the EEX Market Data Hub and writes one CSV per contract:

    data/raw/eex_hub/table_data/NATGAS/TTF/<Futures|Spot|Indices>/<product>/
        [<maturity type>/]<contract>.csv

with one row per trade date (``tradeDate``, ``deliveryStart``, ``settlPx``, volumes, open
interest). The public API only serves a rolling window (about 31 sessions for futures, 40 days
for spot, one year for indices), so the history builds up by re-running the scraper; nothing is
committed (data/raw is gitignored, EEX data is not redistributed).

This module turns that tree into a panel of monthly forward settlements (trade date x delivery
month), a :class:`~gasstorage.data.forward_curve.ForwardCurve` for any trade date, and spot /
index series.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gasstorage.data.forward_curve import ForwardCurve

DEFAULT_ROOT = Path("data/raw/eex_hub")
_AREA = Path("table_data") / "NATGAS" / "TTF"


def _require(root: Path) -> Path:
    base = Path(root) / _AREA
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found: run the eex-scraper (python run.py run) with its output set to "
            f"{DEFAULT_ROOT} (see data/README.md)"
        )
    return base


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["tradeDate"])
    return df[pd.to_numeric(df["settlPx"], errors="coerce").notna()]


def futures_panel(
    root: str | Path = DEFAULT_ROOT, product: str = "Physical", maturity_type: str = "Month"
) -> pd.DataFrame:
    """Settlement prices, index = trade date, columns = delivery month (``PeriodIndex``)."""
    folder = _require(Path(root)) / "Futures" / product / maturity_type
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no {product} {maturity_type} futures in {folder}")
    data = pd.concat([_read(f) for f in files], ignore_index=True)
    data["month"] = pd.PeriodIndex(pd.to_datetime(data["deliveryStart"]), freq="M")
    panel = data.pivot_table(index="tradeDate", columns="month", values="settlPx", aggfunc="last")
    panel = panel.reindex(
        columns=pd.period_range(panel.columns.min(), panel.columns.max(), freq="M")
    )
    panel.index.name = "trade_date"
    panel.columns.name = "delivery_month"
    return panel.sort_index()


def curve_on(
    root: str | Path, trade_date: str | pd.Timestamp, product: str = "Physical"
) -> ForwardCurve:
    """Monthly forward curve settled on ``trade_date`` (delivery months must be contiguous)."""
    panel = futures_panel(root, product)
    day = pd.Timestamp(trade_date)
    if day not in panel.index:
        raise ValueError(f"no {product} settlements on {day.date()} in {root}")
    row = panel.loc[day]
    first, last = row.first_valid_index(), row.last_valid_index()
    window = row.loc[first:last]
    if window.isna().any():
        missing = [str(m) for m in window.index[window.isna()]]
        raise ValueError(f"settlements on {day.date()} are not contiguous (missing {missing})")
    return ForwardCurve(
        months=pd.PeriodIndex(window.index, freq="M"),
        prices=window.to_numpy(dtype=float),
        source=f"EEX hub {product} month futures, settlements of {day.date()}",
        is_synthetic=False,
    )


def series(root: str | Path, short_code: str) -> pd.Series:
    """Daily settlement series of a spot or index contract, by short code (e.g. ``TTFDA``)."""
    base = _require(Path(root))
    stem = short_code.replace(" ", "_")
    names = (stem, f"{stem}_SPOT", f"{stem}_INDEX")
    matches = sorted(p for p in base.rglob("*.csv") if p.stem in names)
    if not matches:
        raise FileNotFoundError(f"no file for {short_code!r} under {base}")
    df = _read(matches[0]).sort_values("tradeDate")
    out = pd.Series(
        df["settlPx"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["tradeDate"], name="trade_date"),
        name=short_code,
    )
    return out[~out.index.duplicated(keep="last")]
