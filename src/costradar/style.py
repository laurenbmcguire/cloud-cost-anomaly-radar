"""Single matplotlib style helper for every chart in the repo (brand palette)."""

from __future__ import annotations

from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BACKGROUND = "#FFFBF6"
INK = "#241A18"
MUTED = "#5E4C46"
GRID = "#EAE0D5"
CRIMSON = "#BE1E3C"
DEEP_CRIMSON = "#8E1630"
GOLD = "#B08A4E"
SAND = "#E3D2B4"

SERIES_COLORS = (CRIMSON, GOLD, DEEP_CRIMSON, SAND, MUTED)
PROVIDER_COLORS = {"AWS": DEEP_CRIMSON, "Azure": CRIMSON, "GCP": GOLD}

DPI = 160
FIGSIZE = (10.0, 5.625)  # 1600 x 900 px at 160 dpi
TITLE_SIZE = 15


def _font_family() -> list[str]:
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [name for name in ("Inter", "DejaVu Sans") if name in available]
    return preferred or ["sans-serif"]


def apply_style() -> None:
    """Apply the brand rcParams globally."""
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": _font_family(),
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": MUTED,
            "axes.titlesize": TITLE_SIZE,
            "axes.titlelocation": "left",
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "axes.prop_cycle": matplotlib.cycler(color=list(SERIES_COLORS)),
        }
    )


def new_figure(nrows: int = 1, ncols: int = 1, **kwargs):
    apply_style()
    return plt.subplots(nrows, ncols, figsize=FIGSIZE, **kwargs)


def subtitle(ax, text: str) -> None:
    """Muted one-line subtitle under the left-aligned title."""
    ax.text(0.0, 1.015, text, transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom", ha="left")


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BACKGROUND)
    plt.close(fig)
    return path
