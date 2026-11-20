"""Tests for the reader of the EEX Market Data Hub scraper output (local CSV tree)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gasstorage.data import eex_hub

# Column layout written by the scraper (one CSV per contract, one row per trade date).
COLUMNS = [
    "tradeDate", "shortCode", "maturityDate", "maturity", "maturityType", "deliveryStart", "tenor",
    "commodity", "pricing", "area", "product", "productSpecific", "settlPx", "currency",
    "totVolTrdd", "uOM", "grossOpenInt", "grossOpenIntSz", "netOpenInt", "netOpenIntSz",
    "deliveryDay", "scrapedAt",
]  # fmt: skip


def _row(trade: str, maturity: str, start: str, px: float | None, mtype: str = "Month") -> dict:
    return {
        "tradeDate": trade, "shortCode": "G3BM", "maturityDate": maturity, "maturity": maturity,
        "maturityType": mtype, "deliveryStart": start, "tenor": start[:7], "commodity": "NATGAS",
        "pricing": "F", "area": "TTF", "product": "Physical", "productSpecific": "",
        "settlPx": px, "currency": "EUR", "totVolTrdd": 0, "uOM": "MWh", "grossOpenInt": 0,
        "grossOpenIntSz": 0, "netOpenInt": "", "netOpenIntSz": "", "deliveryDay": "",
        "scrapedAt": "2026-11-20T10:24:27+00:00",
    }  # fmt: skip


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


@pytest.fixture
def hub(tmp_path: Path) -> Path:
    month_dir = tmp_path / "table_data" / "NATGAS" / "TTF" / "Futures" / "Physical" / "Month"
    for ym, start, prices in [
        ("202704", "2027-04-01", (71.0, 71.5)),
        ("202705", "2027-05-01", (67.0, None)),
        ("202706", "2027-06-01", (65.0, 65.2)),
    ]:
        _write(
            month_dir / f"G3BM_{ym}.csv",
            [
                _row("2026-11-16", ym, start, prices[0]),
                _row("2026-11-17", ym, start, prices[1]),
            ],
        )
    # real naming: <shortCode>_SPOT.csv for spot, <shortCode with underscores>_INDEX.csv for indices
    spot = tmp_path / "table_data" / "NATGAS" / "TTF" / "Spot" / "DA" / "TTFDA_SPOT.csv"
    rows = [_row(d, "", "", px, mtype="") for d, px in [("2026-11-16", 80.0), ("2026-11-17", 82.5)]]
    for r in rows:
        r.update(shortCode="TTFDA", pricing="S", product="DA")
    _write(spot, rows)
    return tmp_path


def test_futures_panel_has_one_column_per_delivery_month(hub: Path) -> None:
    panel = eex_hub.futures_panel(hub)
    assert list(panel.index.strftime("%Y-%m-%d")) == ["2026-11-16", "2026-11-17"]
    assert list(panel.columns.strftime("%Y-%m")) == ["2027-04", "2027-05", "2027-06"]
    assert panel.loc["2026-11-16", pd.Period("2027-05", "M")] == pytest.approx(67.0)
    assert np.isnan(panel.loc["2026-11-17", pd.Period("2027-05", "M")])  # missing settlement


def test_curve_on_a_complete_date_is_a_market_forward_curve(hub: Path) -> None:
    curve = eex_hub.curve_on(hub, "2026-11-16")
    np.testing.assert_allclose(curve.prices, [71.0, 67.0, 65.0])
    assert not curve.is_synthetic
    assert "2026-11-16" in curve.source


def test_curve_on_a_date_with_a_gap_is_rejected(hub: Path) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        eex_hub.curve_on(hub, "2026-11-17")


def test_curve_on_unknown_date_is_rejected(hub: Path) -> None:
    with pytest.raises(ValueError, match="2026-11-18"):
        eex_hub.curve_on(hub, "2026-11-18")


def test_series_reads_spot_by_short_code(hub: Path) -> None:
    s = eex_hub.series(hub, "TTFDA")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-11-16", "2026-11-17"]
    np.testing.assert_allclose(s.to_numpy(), [80.0, 82.5])


def test_missing_hub_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eex-scraper"):
        eex_hub.futures_panel(tmp_path / "nowhere")


def test_config_source_eex_hub_uses_the_valuation_date(hub: Path) -> None:
    import datetime as dt

    from gasstorage.config import ForwardCurveConfig, SyntheticCurveConfig
    from gasstorage.data.forward_curve import ForwardCurve

    syn = SyntheticCurveConfig(30.0, 5.0, 1.0)
    cfg = ForwardCurveConfig(source="eex_hub", csv_path=hub, synthetic=syn)
    curve = ForwardCurve.from_config(
        cfg, dt.date(2027, 4, 1), dt.date(2027, 6, 30), as_of=dt.date(2026, 11, 16)
    )
    np.testing.assert_allclose(curve.prices, [71.0, 67.0, 65.0])
    with pytest.raises(ValueError, match="as_of"):
        ForwardCurve.from_config(cfg, dt.date(2027, 4, 1), dt.date(2027, 6, 30))
