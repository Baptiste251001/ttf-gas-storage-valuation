r"""Monthly forward curve $F(0, T)$: CSV loader, synthetic generator and daily interpolation.

The curve is a vector of monthly delivery prices (EUR/MWh). Daily prices are obtained by
piecewise-constant interpolation: every gas day $t$ in delivery month $m(t)$ gets
$F(0, t) = F_{m(t)}$, consistent with a month contract delivering a flat daily volume.

Two sources are supported:

* **CSV** (``delivery_month, price``): settlement prices typed by hand from ICE Endex or EEX.
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
    def from_config(cls, cfg: ForwardCurveConfig, start: dt.date, end: dt.date) -> ForwardCurve:
        """Build the curve for a contract delivering from ``start`` to ``end`` (inclusive)."""
        first = pd.Period(start, freq="M")
        last = pd.Period(end, freq="M")
        if cfg.source == "synthetic":
            n = (last - first).n + 1
            return cls.synthetic(first, n, cfg.synthetic)
        assert cfg.csv_path is not None  # guaranteed by ForwardCurveConfig validation
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

    def to_frame(self) -> pd.DataFrame:
        """Curve as a DataFrame with the CSV column names."""
        return pd.DataFrame({"delivery_month": self.months.strftime("%Y-%m"), "price": self.prices})
