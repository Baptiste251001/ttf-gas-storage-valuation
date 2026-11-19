"""Time conventions shared by the price model and the valuation.

Model time is measured in years from the valuation date with the ACT/365 convention:
$\\tau(d) = (d - d_0) / 365$ where $d - d_0$ is a number of calendar days.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def year_fraction(start: dt.date, dates: pd.DatetimeIndex) -> np.ndarray:
    """ACT/365 year fractions from ``start`` to each date."""
    days = (pd.DatetimeIndex(dates) - pd.Timestamp(start)).days.to_numpy(dtype=float)
    return days / DAYS_PER_YEAR
