"""One stage loader + normalizer, replacing 4 copy-pasted implementations
across prep_figure_{0,1,2,4}.py.

The bodies were confirmed identical in prep_figure_1/2/4 (differently named:
`_normalize_stage` x2, `_normalize_major_stage` x1). prep_figure_0's version
was a narrower boolean ("matched"/None) built on the exact same regex —
replaced at call sites by `normalize_stage(raw) is not None`, not a separate
function.

prep_figure_4's `_STAGE_IV_TOKEN` is NOT equivalent to
`normalize_stage(raw) == "IV"` — it also matches decimal-then-substage forms
like "4.0A" that `normalize_stage` rejects (the trailing `\\.0+$` strip only
fires when the decimal is at the very end of the string). Kept as a distinct
check, `is_stage_iv`, rather than folded into `normalize_stage`.
"""
import pickle
import re

import pandas as pd

from config import STAGE_PATH

STAGE_ORDER = ["I", "II", "III", "IV"]

_STAGE_TOKEN = re.compile(r"^(IV|III|II|I|4|3|2|1)[A-D]?$")
_STAGE_IV_TOKEN = re.compile(r"^(IV|4(\.0+)?)[A-D]?$", re.IGNORECASE)
_ARABIC_TO_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}


def normalize_stage(raw) -> str | None:
    """Map a raw stage value to a major stage in {I, II, III, IV}, collapsing
    substages (IVA -> IV), arabic numerals (4 -> IV), and float repr (2.0 -> II).
    Returns None for missing / unknown / in-situ / unstageable values."""
    if pd.isna(raw):
        return None
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    s = re.sub(r"\.0+$", "", s)
    m = _STAGE_TOKEN.match(s)
    if not m:
        return None
    token = m.group(1)
    return _ARABIC_TO_ROMAN.get(token, token)


def is_stage_iv(raw) -> bool:
    """True if raw normalizes to stage IV, using the more permissive token
    prep_figure_4 uses for its %-stage-IV metric (also matches e.g. "4.0A")."""
    if pd.isna(raw):
        return False
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    return bool(_STAGE_IV_TOKEN.match(s))


def load_stage_map() -> dict[int, object] | None:
    """Raw DFCI_MRN -> stage value from the derived-stage pickle, or None if
    unreadable (callers fall back to the one-hot cancer_stage_df.csv.gz)."""
    try:
        with open(STAGE_PATH, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
        print(f"  stage pickle unavailable ({type(e).__name__})")
        return None
