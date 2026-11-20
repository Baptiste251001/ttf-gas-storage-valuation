r"""Monthly forward curve $F(0, T)$: CSV loader, synthetic generator and daily interpolation.

The curve is a vector of monthly delivery prices (EUR/MWh). Daily prices are obtained by
piecewise-constant interpolation: every gas day $t$ in delivery month $m(t)$ gets
$F(0, t) = F_{m(t)}$, consistent with a month contract delivering a flat daily volume.

Sources:

* **Exchange settlement table** (``contract, settlement_eur_mwh``) as published by EEX or ICE
  Endex, with contract labels ``Apr-27``, ``Q2-27``, ``SUMMER-27``, ``WINTER-27``, ``CAL-28``.
  Month contracts build the curve; quarter / season / calendar strips are used as a consistency
  check, since a strip price must equal the day-weighted average of its months:

  .. math::
     F_{	ext{strip}} = rac{\sum_{m \in 	ext{strip}} d_m F_m}{\sum_{m \in 	ext{strip}} d_m},
     \qquad d_m = 	ext{number of days in month } m.

* **Monthly CSV** (``delivery_month, price``), e.g. derived from such a table.
* **Synthetic**: a stylised curve, clearly labelled as such, for development and tests:

  .. math::
     F_m = L + A \cos\!\Big(\frac{2\pi(\text{month}(m) - 1)}{12}\Big) + c \, \frac{k_m}{12},

  where $L$ is the base level, $A$ the winter premium (peak in January, trough in July),
  $c$ the contango in EUR/MWh per year and $k_m$ the number of months since the first one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gasstorage.config import ForwardCurveConfig, SyntheticCurveConfig

CSV_COLUMNS = ("delivery_month", "price")
SETTLEMENT_COLUMNS = ("contract", "settlement_eur_mwh")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def parse_contract(label: str) -> tuple[pd.Period, pd.Period]:
    """First and last delivery month of an exchange contract label.

    ``Apr-27`` (month), ``Q2-27`` (quarter), ``SUMMER-27`` (Apr-Sep 2027), ``WINTER-27``
    (Oct 2027 - Mar 2028, gas season convention), ``CAL-28`` (Jan-Dec 2028).
    """
    try:
        head, yy = label.strip().upper().rsplit("-", 1)
        year = 2000 + int(yy)
    except ValueError:
        raise ValueError(f"unknown contract label {label!r}") from None

    def month(y: int, m: int) -> pd.Period:
        return pd.Period(year=y, month=m, freq="M")

    if head in _MONTHS:
        m = month(year, _MONTHS.index(head) + 1)
        return m, m
    if len(head) == 2 and head[0] == "Q" and head[1] in "1234":
        q = int(head[1])
        return month(year, 3 * q - 2), month(year, 3 * q)
    if head == "SUMMER":
        return month(year, 4), month(year, 9)
    if head == "WINTER":
        return month(year, 10), month(year + 1, 3)
    if head == "CAL":
        return month(year, 1), month(year, 12)
    raise ValueError(f"unknown contract label {label!r}")


def load_settlements(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load an exchange settlement table and add ``first_month``, ``last_month``, ``n_months``."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if not set(SETTLEMENT_COLUMNS).issubset(df.columns):
        raise ValueError(f"expected columns {SETTLEMENT_COLUMNS}, got {list(df.columns)}")
    periods = [parse_contract(c) for c in df["contract"]]
    df["first_month"] = [p[0] for p in periods]
    df["last_month"] = [p[1] for p in periods]
    df["n_months"] = [(b - a).n + 1 for a, b in periods]
    df["settlement_eur_mwh"] = df["settlement_eur_mwh"].astype(float)
    return df


def strip_consistency(curve: ForwardCurve, table: pd.DataFrame) -> pd.DataFrame:
    """Compare multi-month strip settlements with the day-weighted average of the month curve.

    Only strips whose months are all on the curve are reported.
    """
    rows = []
    for rec in table[table["n_months"] > 1].itertuples(index=False):
        if rec.first_month < curve.months[0] or rec.last_month > curve.months[-1]:
            continue
        implied = curve.strip_price(rec.first_month, rec.last_month)
        rows.append(
            {
                "contract": rec.contract,
                "settlement_eur_mwh": rec.settlement_eur_mwh,
                "implied_eur_mwh": implied,
                "diff_eur_mwh": rec.settlement_eur_mwh - implied,
            }
        )
    return pd.DataFrame(
        rows, columns=["contract", "settlement_eur_mwh", "implied_eur_mwh", "diff_eur_mwh"]
    )


def _validate(months: pd.PeriodIndex, prices: np.ndarray) -> None:
    if len(months) == 0:
        raise ValueError("forward curve is empty")
    if months.has_duplicates:
        raise ValueError("forward curve has duplicate delivery months")
    ordinals = months.asi8
    if np.any(np.diff(ordinals) != 1):
        raise ValueError("forward curve delivery months must be contiguous")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("forward curve prices must be finite and positive")


@dataclass(frozen=True)
class ForwardCurve:
    """Monthly forward curve with contiguous delivery months.

    Attributes:
        months: contiguous monthly ``PeriodIndex`` of delivery months, ascending.
        prices: forward prices in EUR/MWh, aligned with ``months``.
        source: human-readable provenance (file path or ``"SYNTHETIC ..."``).
        is_synthetic: ``True`` if the curve is not market data.
    """

    months: pd.PeriodIndex
    prices: np.ndarray
    source: str
    is_synthetic: bool

    def __post_init__(self) -> None:
        _validate(self.months, self.prices)
        self.prices.setflags(write=False)

    # -- constructors ------------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> ForwardCurve:
        """Load a curve from a CSV with columns ``delivery_month`` (e.g. 2027-04) and ``price``."""
        path = Path(path)
        df = pd.read_csv(path)
        if not set(CSV_COLUMNS).issubset(df.columns):
            raise ValueError(f"{path}: expected columns {CSV_COLUMNS}, got {list(df.columns)}")
        months = pd.PeriodIndex(pd.to_datetime(df["delivery_month"]), freq="M")
        prices = df["price"].to_numpy(dtype=float)
        if months.has_duplicates:
            raise ValueError(f"{path}: duplicate delivery months")
        order = np.argsort(months.asi8)
        return cls(months[order], prices[order], source=str(path), is_synthetic=False)

    @classmethod
    def from_settlements(cls, table: pd.DataFrame, source: str) -> ForwardCurve:
        """Build the curve from the month contracts of a :func:`load_settlements` table."""
        months_only = table[table["n_months"] == 1].sort_values("first_month")
        months = pd.PeriodIndex(months_only["first_month"].tolist(), freq="M")
        prices = months_only["settlement_eur_mwh"].to_numpy(dtype=float)
        return cls(months, prices, source=source, is_synthetic=False)

    @classmethod
    def synthetic(
        cls, start: pd.Period, n_months: int, params: SyntheticCurveConfig
    ) -> ForwardCurve:
        """Generate the SYNTHETIC seasonal curve defined in the module docstring."""
        months = pd.period_range(start=start, periods=n_months, freq="M")
        k = np.arange(n_months, dtype=float)
        season = np.cos(2.0 * np.pi * (months.month.to_numpy() - 1) / 12.0)
        prices = (
            params.base_level_eur_mwh
            + params.winter_premium_eur_mwh * season
            + params.contango_eur_mwh_per_year * k / 12.0
        )
        if np.any(prices <= 0):
            raise ValueError("synthetic curve parameters produce non-positive prices")
        source = (
            f"SYNTHETIC (L={params.base_level_eur_mwh}, A={params.winter_premium_eur_mwh}, "
            f"c={params.contango_eur_mwh_per_year})"
        )
        return cls(months, prices, source=source, is_synthetic=True)

    @classmethod
    def from_config(
        cls,
        cfg: ForwardCurveConfig,
        start: dt.date,
        end: dt.date,
        as_of: dt.date | None = None,
    ) -> ForwardCurve:
        """Build the curve for a contract delivering from ``start`` to ``end`` (inclusive)."""
        first = pd.Period(start, freq="M")
        last = pd.Period(end, freq="M")
        if cfg.source == "synthetic":
            n = (last - first).n + 1
            return cls.synthetic(first, n, cfg.synthetic)
        assert cfg.csv_path is not None  # guaranteed by ForwardCurveConfig validation
        if cfg.source == "eex_hub":
            if as_of is None:
                raise ValueError("source 'eex_hub' needs the trade date: pass as_of")
            from gasstorage.data import eex_hub  # local import: eex_hub imports this module

            curve = eex_hub.curve_on(cfg.csv_path, pd.Timestamp(as_of))
            if curve.months[0] > first or curve.months[-1] < last:
                raise ValueError(f"EEX hub curve of {as_of} does not cover {first}..{last}")
            return curve
        columns = pd.read_csv(cfg.csv_path, nrows=0).columns
        if set(SETTLEMENT_COLUMNS).issubset(columns):
            table = load_settlements(cfg.csv_path)
            curve = cls.from_settlements(table, source=f"exchange settlements: {cfg.csv_path}")
        else:
            curve = cls.from_csv(cfg.csv_path)
        if curve.months[0] > first or curve.months[-1] < last:
            raise ValueError(
                f"{cfg.csv_path}: curve {curve.months[0]}..{curve.months[-1]} does not cover "
                f"the contract period {first}..{last}"
            )
        return curve

    # -- accessors ---------------------------------------------------------------------------

    def daily(self, dates: pd.DatetimeIndex) -> np.ndarray:
        r"""Piecewise-constant daily prices $F(0, t) = F_{m(t)}$ for each date."""
        idx = pd.DatetimeIndex(dates).to_period("M").asi8 - self.months.asi8[0]
        if np.any(idx < 0) or np.any(idx >= len(self.months)):
            raise ValueError(f"dates outside the curve range {self.months[0]}..{self.months[-1]}")
        return self.prices[idx]

    def strip_price(self, first: pd.Period, last: pd.Period) -> float:
        """Day-weighted average price of the delivery months ``first..last`` (inclusive)."""
        mask = (self.months >= first) & (self.months <= last)
        if mask.sum() != (last - first).n + 1:
            raise ValueError(f"strip {first}..{last} is not fully covered by the curve")
        days = self.months[mask].days_in_month.to_numpy(dtype=float)
        return float(np.dot(days, self.prices[mask]) / days.sum())

    def to_frame(self) -> pd.DataFrame:
        """Curve as a DataFrame with the CSV column names."""
        return pd.DataFrame({"delivery_month": self.months.strftime("%Y-%m"), "price": self.prices})
