"""Tests for the shared plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from gasstorage import plotting


def test_save_figure_writes_png_and_returns_path(tmp_path: Path) -> None:
    plotting.apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = plotting.save_figure(fig, "demo", directory=tmp_path / "figs")
    assert path == tmp_path / "figs" / "demo.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_synthetic_label_is_stamped_on_axes() -> None:
    fig, ax = plt.subplots()
    plotting.mark_synthetic(ax)
    texts = [t.get_text() for t in ax.texts]
    assert any("SYNTHETIC" in t for t in texts)
    plt.close(fig)
