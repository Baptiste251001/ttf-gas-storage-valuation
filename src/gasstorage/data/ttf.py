"""Historical TTF front-month prices.

Primary source: Yahoo Finance continuous front-month contract ``TTF=F`` (EUR/MWh) through
``yfinance``. Yahoo stitches successive front-month contracts without roll adjustment, so the
series has small jumps at expiries; this matters for return statistics, not for the level-based
mean-reversion calibration used in this project. Yahoo data is for personal use and is never
committed: it is cached under ``data/raw/`` (gitignored).

Fallback: any CSV with columns ``date, price_eur_mwh`` (e.g. exported from ICE Endex / EEX
settlement history), loaded with :func:`load_ttf_csv`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

TICKER = "TTF=F"
CSV_COLUMNS = ("date", "price_eur_mwh")
DEFAULT_CACHE = Path("data/raw/ttf_front_month.csv")


def _validate(series: pd.Series, origin: str) -> pd.Series:
    if series.index.has_duplicates:
        raise ValueError(f"{origin}: duplicate dates")
    values = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{origin}: prices must be finite and positive")
    return series


def load_ttf_csv(path: str | Path) -> pd.Series:
    """Load a TTF price history CSV (``date, price_eur_mwh``) as a sorted daily series."""
    path = Path(path)
    df = pd.read_csv(path)
    if not set(CSV_COLUMNS).issubset(df.columns):
        raise ValueError(f"{path}: expected columns {CSV_COLUMNS}, got {list(df.columns)}")
    index = pd.DatetimeIndex(pd.to_datetime(df["date"]), name="date")
    series = pd.Series(df["price_eur_mwh"].to_numpy(dtype=float), index=index, name=CSV_COLUMNS[1])
    return _validate(series.sort_index(), str(path))


def close_from_yfinance(frame: pd.DataFrame, ticker: str = TICKER) -> pd.Series:
    """Extract the daily close from a ``yfinance.download`` frame, dropping empty rows."""
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):  # MultiIndex columns (field, ticker)
        close = close[ticker]
    close = close.dropna()
    close = close[close > 0]
    series = pd.Series(
        close.to_numpy(dtype=float),
        index=pd.DatetimeIndex(close.index, name="date"),
        name=CSV_COLUMNS[1],
    )
    return _validate(series.sort_index(), f"yfinance {ticker}")


def download_ttf_yfinance(start: dt.date = dt.date(2017, 1, 1)) -> pd.Series:
    """Download the TTF front-month close from Yahoo Finance (requires network + ``yfinance``)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("yfinance is not installed: pip install -e '.[data]'") from exc
    frame = yf.download(TICKER, start=start.isoformat(), progress=False, auto_adjust=False)
    if frame is None or frame.empty:
        raise RuntimeError(
            f"yfinance returned no data for {TICKER}. Download a CSV with columns "
            f"{CSV_COLUMNS} and load it with load_ttf_csv()."
        )
    return close_from_yfinance(frame)


def get_ttf_history(
    cache_path: str | Path = DEFAULT_CACHE,
    refresh: bool = False,
    downloader: Callable[[], pd.Series] = download_ttf_yfinance,
) -> pd.Series:
    """Return the cached TTF history, downloading and caching it if missing or ``refresh``."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return load_ttf_csv(cache_path)
    series = downloader()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.rename_axis("date").to_frame().to_csv(cache_path, date_format="%Y-%m-%d")
    return series
