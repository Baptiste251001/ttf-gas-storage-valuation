"""CLI entry point.

Usage: ``python scripts/run_valuation.py --config configs/storage_default.yaml`` (see ``--help``).
"""

from gasstorage.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
