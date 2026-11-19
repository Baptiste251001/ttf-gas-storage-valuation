"""Shared matplotlib style for notebooks and report figures.

Colors follow a validated colorblind-safe categorical order; series hues are assigned in this
fixed order and never cycled. Text uses neutral ink, never the series color. Figures with
synthetic inputs carry a visible ``SYNTHETIC`` stamp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL_FILL = "#f0efec"
FIGURES_DIR = Path("reports/figures")


def apply_style() -> None:
    """Set the project-wide matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.figsize": (9, 4.5),
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        }
    )


def mark_synthetic(ax: Axes, text: str = "SYNTHETIC DATA", corner: str = "upper right") -> None:
    """Stamp an axes as showing synthetic (non-market) data, in a free ``corner``."""
    vert, horiz = corner.split()
    ax.text(
        0.99 if horiz == "right" else 0.01,
        0.97 if vert == "upper" else 0.03,
        text,
        transform=ax.transAxes,
        ha=horiz,
        va="top" if vert == "upper" else "bottom",
        fontsize=9,
        fontweight="bold",
        color=TEXT_SECONDARY,
        bbox={"boxstyle": "round,pad=0.3", "fc": NEUTRAL_FILL, "ec": GRID},
    )


def save_figure(fig: Figure, name: str, directory: str | Path = FIGURES_DIR) -> Path:
    """Save ``fig`` as ``<directory>/<name>.png`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
