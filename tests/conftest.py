"""Shared test helpers.

Market-data tests need the EEX settlement table referenced by ``configs/storage_default.yaml``.
It lives in ``data/raw/`` (gitignored, see ``data/README.md``), so these tests are skipped on a
fresh clone or in CI instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gasstorage.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MARKET_CONFIG = ROOT / "configs" / "storage_default.yaml"


def market_data_available() -> bool:
    path = load_config(MARKET_CONFIG).market.forward_curve.csv_path
    return path is not None and (ROOT / path).exists()


def skip_without_market_data() -> None:
    if not market_data_available():
        pytest.skip("market forward curve CSV not available (see data/README.md)")
