"""Shared utilities for manuscript figure notebooks.

Conventions
-----------
- Every panel is saved as a standalone PNG via `save_panel(fig, "fig2c")`.
- Output goes to FIGURE_OUT_DIR (see config.py).
- Color palettes are kept consistent across figures (modality palette spans Fig 2-3;
  cluster palette is used in Fig 4).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from config import FIGURE_DATA_DIR, FIGURE_OUT_DIR

logger = logging.getLogger(__name__)

FIGURE_OUT_DIR = Path(FIGURE_OUT_DIR)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_panel(fig: plt.Figure, name: str) -> Path:
    """Save `fig` as `<FIGURE_OUT_DIR>/<name>.png` and return the path.

    `name` should be like 'fig2c' (no extension). PDFs are explicitly disabled.
    """
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_OUT_DIR / f"{name}.png"
    fig.savefig(out, format="png")
    print(f"[saved] {out}")
    return out


# ---------------------------------------------------------------------------
# Figure-data IO (prep scripts write, notebooks read)
# ---------------------------------------------------------------------------
def figure_data_path(name: str) -> str:
    """Resolve a filename like 'fig2_metrics.csv' to its absolute path."""
    return os.path.join(FIGURE_DATA_DIR, name)


def save_figure_data(df: pl.DataFrame, name: str) -> str:
    """Write a prepared CSV to FIGURE_DATA_DIR and return the path."""
    if len(df.columns) == 0:
        raise ValueError(f"Refusing to write schema-less figure data for {name}")
    if len(df) == 0:
        # Not raised: many prep functions deliberately return `pd.DataFrame(columns=[...])`
        # on a data-not-available path, and that 0-row CSV is a legitimate output today.
        # But it is currently indistinguishable on disk from a successful run (R turns a
        # missing/empty file into an empty tibble and silently renders a placeholder panel),
        # so at minimum this must not pass without a loud signal in the run log.
        logger.warning("[EMPTY FIGURE DATA] %s has columns but 0 rows -- a downstream panel will likely be a placeholder.", name)
    os.makedirs(FIGURE_DATA_DIR, exist_ok=True)
    out = figure_data_path(name)
    df.write_csv(out)
    print(f"[wrote] {out}  ({len(df):,} rows)")
    return out


def load_figure_data(name: str) -> pl.DataFrame:
    """Read a prepared CSV from FIGURE_DATA_DIR."""
    return pl.read_csv(figure_data_path(name))
