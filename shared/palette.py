"""Loads shared/palette.json — the one definition of manuscript-figure colors,
read identically by Python (here) and R (R/figure_utils.R via jsonlite::fromJSON).
Previously these were two independent literals (_figure_utils.py vs
R/figure_utils.R) with no mechanism to catch drift between them.
"""
import json
from pathlib import Path

_PALETTE_PATH = Path(__file__).resolve().parent / "palette.json"
_palette = json.loads(_PALETTE_PATH.read_text())

MODALITY_ORDER: list[str] = _palette["MODALITY_ORDER"]
MODALITY_COLORS: dict[str, str] = _palette["MODALITY_COLORS"]
MODALITY_DISPLAY: dict[str, str] = _palette["MODALITY_DISPLAY"]
MODEL_COLORS: dict[str, str] = _palette["MODEL_COLORS"]
CLUSTER_COLORS: list[str] = _palette["CLUSTER_COLORS"]
