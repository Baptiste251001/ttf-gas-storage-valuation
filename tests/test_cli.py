"""Tests for the command-line valuation entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gasstorage.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_writes_consistent_json(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    code = main(
        [
            "--config",
            "configs/storage_synthetic.yaml",
            "--paths",
            "300",
            "--levels",
            "26",
            "--json",
            str(out),
        ]
    )
    assert code == 0
    res = json.loads(out.read_text())
    v = res["valuation"]
    assert v["lsmc_value_eur"] > 0 and v["lsmc_se_eur"] > 0
    assert v["extrinsic_eur"] == v["lsmc_value_eur"] - v["intrinsic_same_grid_eur"]
    assert res["settings"]["n_paths_valuation"] == 300
    assert res["curve"]["is_synthetic"] is True


def test_script_runs_as_documented() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_valuation.py",
            "--config",
            "configs/storage_synthetic.yaml",
            "--paths",
            "200",
            "--levels",
            "26",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "LSMC value" in proc.stdout
    assert "lower bound" in proc.stdout
