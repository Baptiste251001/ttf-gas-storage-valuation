"""Time conventions shared by the price model and the valuation.

Model time is measured in years from the valuation date with the ACT/365 convention:
$\\tau(d) = (d - d_0) / 365$ where $d - d_0$ is a number of calendar days.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def year_fraction(start: dt.date, dates: pd.DatetimeIndex) -> np.ndarray:
    """ACT/365 year fractions from ``start`` to each date."""
    days = (pd.DatetimeIndex(dates) - pd.Timestamp(start)).days.to_numpy(dtype=float)
    return days / DAYS_PER_YEAR


@dataclass(frozen=True)
class DecisionGrid:
    """Decision dates of the storage problem.

    ``dates[k]`` is the first gas day of step ``k`` and ``days[k]`` its length in days. Volumes
    moved during step ``k`` are traded at the price of ``dates[k]``.
    """

    dates: pd.DatetimeIndex
    days: np.ndarray


def decision_grid(start: dt.date, end: dt.date, step: str) -> DecisionGrid:
    """Daily or weekly steps from ``start`` to ``end`` (inclusive); the last week may be short."""
    if step not in ("daily", "weekly"):
        raise ValueError(f"unknown step {step!r}")
    length = 1 if step == "daily" else 7
    all_days = pd.date_range(start, end, freq="D")
    dates = all_days[::length]
    stops = dates[1:].append(pd.DatetimeIndex([all_days[-1] + pd.Timedelta(days=1)]))
    days = (stops - dates).days.to_numpy(dtype=float)
    return DecisionGrid(dates=dates, days=days)
