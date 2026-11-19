"""Tests for the TTF history and GIE AGSI+ loaders (no network access)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gasstorage.data import agsi, ttf

# --- TTF history --------------------------------------------------------------------------


def _write_ttf(path: Path, rows: list[tuple[str, float]]) -> Path:
    pd.DataFrame(rows, columns=["date", "price_eur_mwh"]).to_csv(path, index=False)
    return path


def test_ttf_csv_returns_sorted_daily_series(tmp_path: Path) -> None:
    path = _write_ttf(tmp_path / "t.csv", [("2024-01-03", 30.5), ("2024-01-02", 31.0)])
    s = ttf.load_ttf_csv(path)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    np.testing.assert_allclose(s.to_numpy(), [31.0, 30.5])
    assert s.name == "price_eur_mwh"


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([("2024-01-02", 31.0), ("2024-01-02", 30.0)], "duplicate"),
        ([("2024-01-02", 31.0), ("2024-01-03", 0.0)], "positive"),
    ],
)
def test_ttf_csv_rejects_bad_data(
    tmp_path: Path, rows: list[tuple[str, float]], match: str
) -> None:
    path = _write_ttf(tmp_path / "t.csv", rows)
    with pytest.raises(ValueError, match=match):
        ttf.load_ttf_csv(path)


def test_ttf_csv_rejects_wrong_columns(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    pd.DataFrame({"Date": ["2024-01-02"], "Close": [30.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="price_eur_mwh"):
        ttf.load_ttf_csv(path)


def test_close_is_extracted_from_yfinance_frame() -> None:
    # Mirrors yfinance.download output: MultiIndex columns (field, ticker), DatetimeIndex 'Date'.
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="Date")
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    cols = pd.MultiIndex.from_product([fields, ["TTF=F"]], names=["Price", "Ticker"])
    data = np.array(
        [
            [31.0, 31.0, 32.0, 30.0, 30.5, 10],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0],
            [29.0, 29.0, 30.0, 28.0, 29.5, 12],
        ]
    )
    frame = pd.DataFrame(data, index=idx, columns=cols)
    s = ttf.close_from_yfinance(frame, "TTF=F")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]
    np.testing.assert_allclose(s.to_numpy(), [31.0, 29.0])
    assert s.index.name == "date"


def test_cached_history_is_read_without_network(tmp_path: Path) -> None:
    cache = _write_ttf(tmp_path / "t.csv", [("2024-01-02", 31.0)])

    def _no_download(*_: Any, **__: Any) -> pd.Series:
        raise AssertionError("must not download when cache exists")

    s = ttf.get_ttf_history(cache, downloader=_no_download)
    assert s.iloc[0] == 31.0


def test_missing_cache_downloads_and_writes_csv(tmp_path: Path) -> None:
    cache = tmp_path / "raw" / "t.csv"
    fresh = pd.Series(
        [31.0, 30.0],
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date"),
        name="price_eur_mwh",
    )
    s = ttf.get_ttf_history(cache, downloader=lambda: fresh)
    assert cache.exists()
    pd.testing.assert_series_equal(ttf.load_ttf_csv(cache), s, check_freq=False)


# --- GIE AGSI+ -----------------------------------------------------------------------------

# Record copied from the GIE "API access to AGSI / ALSI" manual v007, section 7 (Germany,
# 2022-03-31). Values are strings and use a decimal comma in this example.
RECORD_DE = {
    "name": "Germany",
    "code": "DE",
    "consumption": "995,2735",
    "consumptionFull": "6.33",
    "url": "DE",
    "gasDayStart": "2022-03-31",
    "gasInStorage": "63,8469",
    "injection": "563,62",
    "withdrawal": "591,99",
    "workingGasVolume": "243,4311",
    "injectionCapacity": "3943,99",
    "withdrawalCapacity": "6724,93",
    "status": "C",
    "trend": "-0.01",
    "full": "26.27",
    "info": [],
}


def test_records_are_parsed_to_numeric_frame() -> None:
    df = agsi.parse_records([RECORD_DE])
    row = df.iloc[0]
    assert row["gas_day"] == pd.Timestamp("2022-03-31")
    assert row["country"] == "DE"
    assert row["gas_in_storage_twh"] == pytest.approx(63.8469)
    assert row["working_gas_volume_twh"] == pytest.approx(243.4311)
    assert row["full_pct"] == pytest.approx(26.27)
    assert row["injection_gwh_d"] == pytest.approx(563.62)
    assert row["withdrawal_gwh_d"] == pytest.approx(591.99)
    assert row["status"] == "C"


def test_missing_values_become_nan() -> None:
    rec = {**RECORD_DE, "gasInStorage": "-", "full": "", "injection": None}
    row = agsi.parse_records([rec]).iloc[0]
    assert np.isnan(row["gas_in_storage_twh"])
    assert np.isnan(row["full_pct"])
    assert np.isnan(row["injection_gwh_d"])


def test_dot_decimals_are_parsed() -> None:
    row = agsi.parse_records([{**RECORD_DE, "gasInStorage": "853.4219"}]).iloc[0]
    assert row["gas_in_storage_twh"] == pytest.approx(853.4219)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    """Stands in for requests.Session: records calls, serves one payload per page."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], headers: dict[str, str], timeout: float):
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        return _FakeResponse(self.pages[int(params["page"]) - 1])


def _page(last_page: int, days: list[str]) -> dict[str, Any]:
    return {
        "last_page": last_page,
        "total": len(days),
        "dataset": "DE",
        "gas_day": days[0],
        "data": [{**RECORD_DE, "gasDayStart": d} for d in days],
    }


def test_fetch_country_follows_pagination_and_sends_key() -> None:
    session = _FakeSession([_page(2, ["2022-03-31", "2022-03-30"]), _page(2, ["2022-03-29"])])
    df = agsi.fetch_country(
        "DE", dt.date(2022, 3, 29), dt.date(2022, 3, 31), api_key="k", session=session, pause=0
    )
    assert [c["params"]["page"] for c in session.calls] == [1, 2]
    assert all(c["headers"]["x-key"] == "k" for c in session.calls)
    assert session.calls[0]["params"]["country"] == "DE"
    assert session.calls[0]["params"]["from"] == "2022-03-29"
    assert session.calls[0]["params"]["to"] == "2022-03-31"
    assert session.calls[0]["url"] == agsi.API_URL
    # sorted ascending by gas day
    assert list(df["gas_day"].dt.strftime("%Y-%m-%d")) == ["2022-03-29", "2022-03-30", "2022-03-31"]


def test_api_error_payload_raises_with_message() -> None:
    # Actual response of https://agsi.gie.eu/api without a valid key (HTTP 200).
    denied = {
        "last_page": 0,
        "total": 0,
        "dataset": "storage ERROR",
        "error": "access denied",
        "message": "Invalid or missing API key",
        "data": [],
    }
    with pytest.raises(agsi.AGSIError, match="Invalid or missing API key"):
        agsi.fetch_country(
            "DE",
            dt.date(2022, 3, 29),
            dt.date(2022, 3, 31),
            api_key="bad",
            session=_FakeSession([denied]),
            pause=0,
        )


def test_missing_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agsi.ENV_VAR, raising=False)
    with pytest.raises(agsi.AGSIKeyMissingError, match="AGSI_API_KEY"):
        agsi.fetch_country("DE", dt.date(2022, 1, 1), dt.date(2022, 1, 2), session=_FakeSession([]))


def test_load_storage_levels_uses_cache_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(agsi.ENV_VAR, raising=False)
    cached = agsi.parse_records([RECORD_DE])
    cached.to_csv(tmp_path / "agsi_DE.csv", index=False)
    df = agsi.load_storage_levels(["DE"], dt.date(2022, 1, 1), dt.date(2022, 12, 31), tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["gas_in_storage_twh"] == pytest.approx(63.8469)


@pytest.mark.parametrize(
    ("day", "year", "day_index"),
    [
        ("2023-04-01", 2023, 0),
        ("2023-04-02", 2023, 1),
        ("2024-03-31", 2023, 365),  # 2024 is a leap year
        ("2024-04-01", 2024, 0),
        ("2023-01-15", 2022, 289),
    ],
)
def test_storage_year_assignment(day: str, year: int, day_index: int) -> None:
    df = pd.DataFrame({"gas_day": pd.to_datetime([day])})
    out = agsi.add_storage_year(df)
    assert out.iloc[0]["storage_year"] == year
    assert out.iloc[0]["day_of_storage_year"] == day_index
