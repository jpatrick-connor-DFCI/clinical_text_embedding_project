"""The time-zero-anchor registry.

Mirrors the `schemes.py` registry pattern. Every survival model anchors at
`t=0` from either first treatment (the original, default behavior) or
genomic-specimen sequencing date (the new sensitivity arm). `treatment`
must resolve to every path and column name
that exists today — backwards compatibility is load-bearing so existing
results and all figure-prep readers keep working untouched.
"""

ANCHORS: dict[str, dict[str, str]] = {
    "treatment": {
        "date_col": "first_treatment_date",
        "note_time_col": "NOTE_TIME_REL_TREATMENT",
        "age_col": "AGE_AT_TREATMENTSTART",
    },
    "sequencing": {
        "date_col": "sequencing_date",
        "note_time_col": "NOTE_TIME_REL_SEQUENCING",
        "age_col": "AGE_AT_SEQUENCING",
    },
}
DEFAULT_ANCHOR = "treatment"


def ensure_anchor(anchor: str) -> str:
    if anchor not in ANCHORS:
        valid = ", ".join(sorted(ANCHORS))
        raise ValueError(f"Unsupported anchor '{anchor}'. Valid options: {valid}")
    return anchor


def date_col(anchor: str) -> str:
    return ANCHORS[ensure_anchor(anchor)]["date_col"]


def note_time_col(anchor: str) -> str:
    return ANCHORS[ensure_anchor(anchor)]["note_time_col"]


def age_col(anchor: str) -> str:
    return ANCHORS[ensure_anchor(anchor)]["age_col"]


def anchor_suffix(anchor: str) -> str:
    """Path/filename suffix for a non-default anchor; empty for `treatment`
    so every existing path stays byte-identical."""
    ensure_anchor(anchor)
    return "" if anchor == DEFAULT_ANCHOR else f"__{anchor}"
