"""GIE AGSI+ aggregated gas storage levels (country level: NL, DE, FR, ...).

Source: Gas Infrastructure Europe, AGSI+ transparency platform, https://agsi.gie.eu.
Data may be reused freely provided GIE AGSI+ is cited as the source (GIE API manual v007).

API facts used here (GIE "API access to AGSI / ALSI" manual v007):

* endpoint ``https://agsi.gie.eu/api``, personal key sent in the ``x-key`` header;
* query parameters ``country``, ``from``, ``to`` (gas day start dates), ``page`` (from 1),
  ``size`` (max 300); the response carries ``last_page`` and a ``data`` array;
* numeric values are returned as strings, sometimes with a decimal comma;
* units: ``gasInStorage`` and ``workingGasVolume`` in TWh, ``injection`` / ``withdrawal``
  in GWh/d, ``full`` in % of working gas volume;
* rate limit: 60 calls per minute.

The key is read from the ``AGSI_API_KEY`` environment variable. Without it, cached CSVs in
``data/raw/`` are still usable and a clear error explains how to obtain a key.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

API_URL = "https://agsi.gie.eu/api"
ENV_VAR = "AGSI_API_KEY"
PAGE_SIZE = 300
DEFAULT_CACHE_DIR = Path("data/raw")

# API field -> (column name, kind)
_FIELDS: dict[str, tuple[str, str]] = {
    "gasDayStart": ("gas_day", "date"),
    "code": ("country", "str"),
    "gasInStorage": ("gas_in_storage_twh", "num"),
    "workingGasVolume": ("working_gas_volume_twh", "num"),
    "full": ("full_pct", "num"),
    "injection": ("injection_gwh_d", "num"),
    "withdrawal": ("withdrawal_gwh_d", "num"),
    "injectionCapacity": ("injection_capacity_gwh_d", "num"),
    "withdrawalCapacity": ("withdrawal_capacity_gwh_d", "num"),
    "status": ("status", "str"),
}
COLUMNS = [col for col, _ in _FIELDS.values()]


class AGSIError(RuntimeError):
    """The AGSI+ API returned an error payload."""


class AGSIKeyMissingError(AGSIError):
    """No API key available."""


class _Session(Protocol):
    def get(
        self, url: str, params: Mapping[str, Any], headers: Mapping[str, str], timeout: float
    ) -> Any: ...


def _to_float(value: Any) -> float:
    """Parse an AGSI numeric string: decimal comma or dot; ``"-"`` / empty / None -> NaN."""
    if value is None:
        return np.nan
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if text in ("", "-"):
        return np.nan
    text = text.replace(",", ".") if "." not in text else text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert raw AGSI ``data`` entries to a typed DataFrame sorted by gas day."""
    rows = []
    for rec in records:
        row: dict[str, Any] = {}
        for field, (col, kind) in _FIELDS.items():
            raw = rec.get(field)
            if kind == "num":
                row[col] = _to_float(raw)
            elif kind == "date":
                row[col] = pd.Timestamp(raw)
            else:
                row[col] = None if raw is None else str(raw)
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["gas_day"] = pd.to_datetime(df["gas_day"])
    return df.sort_values("gas_day", ignore_index=True)


def get_api_key() -> str | None:
    """Return the AGSI+ API key from the environment, or None."""
    key = os.environ.get(ENV_VAR, "").strip()
    return key or None


def _missing_key_error() -> AGSIKeyMissingError:
    return AGSIKeyMissingError(
        f"No AGSI+ API key found. Register for free at https://agsi.gie.eu/account, copy the "
        f"key from your account page and set the {ENV_VAR} environment variable. Cached CSVs "
        f"in data/raw/ can be used without a key."
    )


def fetch_country(
    country: str,
    start: dt.date,
    end: dt.date,
    api_key: str | None = None,
    session: _Session | None = None,
    pause: float = 1.1,
) -> pd.DataFrame:
    """Download daily aggregated storage data for one country between two gas days.

    ``pause`` (seconds) is waited between page requests to stay below 60 calls/minute.
    """
    api_key = api_key or get_api_key()
    if api_key is None:
        raise _missing_key_error()
    if session is None:  # pragma: no cover - real network
        import requests

        session = requests.Session()

    records: list[Mapping[str, Any]] = []
    page, last_page = 1, 1
    while page <= last_page:
        params = {
            "country": country,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "page": page,
            "size": PAGE_SIZE,
        }
        response = session.get(API_URL, params=params, headers={"x-key": api_key}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise AGSIError(f"AGSI+ API error: {payload.get('message') or payload['error']}")
        data = payload.get("data", [])
        records.extend([data] if isinstance(data, Mapping) else data)
        last_page = int(payload.get("last_page") or 0)
        page += 1
        if page <= last_page and pause > 0:
            time.sleep(pause)
    return parse_records(records)


def _cache_file(cache_dir: Path, country: str) -> Path:
    return cache_dir / f"agsi_{country.upper()}.csv"


def load_storage_levels(
    countries: Sequence[str],
    start: dt.date,
    end: dt.date,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Storage levels for several countries, from the CSV cache or the API (then cached).

    Raises :class:`AGSIKeyMissingError` with instructions if a country is not cached and no
    API key is set.
    """
    cache_dir = Path(cache_dir)
    frames = []
    for country in countries:
        path = _cache_file(cache_dir, country)
        if path.exists() and not refresh:
            df = pd.read_csv(path, parse_dates=["gas_day"])
        else:
            df = fetch_country(country, start, end)
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, date_format="%Y-%m-%d")
        mask = (df["gas_day"] >= pd.Timestamp(start)) & (df["gas_day"] <= pd.Timestamp(end))
        frames.append(df.loc[mask])
    return pd.concat(frames, ignore_index=True)


def add_storage_year(df: pd.DataFrame, date_col: str = "gas_day") -> pd.DataFrame:
    """Add ``storage_year`` (year in which the 1 April - 31 March cycle starts) and
    ``day_of_storage_year`` (0 on 1 April)."""
    out = df.copy()
    days = pd.DatetimeIndex(out[date_col])
    year = np.where(days.month >= 4, days.year, days.year - 1)
    start = pd.to_datetime({"year": year, "month": 4, "day": 1})
    out["storage_year"] = year
    out["day_of_storage_year"] = (days - pd.DatetimeIndex(start)).days.to_numpy()
    return out


def fill_profile(
    levels: pd.DataFrame,
    country: str,
    years: list[int] | None = None,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    min_years: int = 1,
) -> pd.DataFrame:
    """Quantiles of the fill level (% of WGV) across storage years, by day of storage year.

    ``levels`` must contain ``storage_year`` and ``day_of_storage_year`` (see
    :func:`add_storage_year`). Days observed in fewer than ``min_years`` storage years (e.g. day
    365, which only leap years have) are dropped. Columns are the requested quantiles.
    """
    sub = levels[levels["country"] == country]
    if years is not None:
        sub = sub[sub["storage_year"].isin(years)]
    grouped = sub.groupby("day_of_storage_year")["full_pct"]
    out = pd.DataFrame({q: grouped.quantile(q) for q in quantiles})
    return out[grouped.count() >= min_years]


def aggregate_fill(levels: pd.DataFrame, countries: list[str], name: str) -> pd.DataFrame:
    """Aggregate several countries into one storage: fill = sum(storage) / sum(WGV), in %."""
    sub = levels[levels["country"].isin(countries)]
    daily = sub.groupby("gas_day")[["gas_in_storage_twh", "working_gas_volume_twh"]].sum()
    out = pd.DataFrame(
        {
            "gas_day": daily.index,
            "country": name,
            "full_pct": 100.0 * daily["gas_in_storage_twh"] / daily["working_gas_volume_twh"],
        }
    ).reset_index(drop=True)
    return out
