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
    # ADT start, for the COMPASS time-to-platinum arm. Unlike the two anchors
    # above, the anchor dates are not carried in this project's cohort frame:
    # they come from COMPASS's TREATMENT_ANCHOR_DATE, joined on DFCI_MRN by
    # COMPASS/data_preprocessing/build_text_embedding_inputs.py. Registered
    # here so anchor_suffix() namespaces any file this project writes for it
    # and so note_time_col() is the single definition of the column name that
    # both repos compute against.
    "adt": {
        "date_col": "adt_start_date",
        "note_time_col": "NOTE_TIME_REL_ADT",
        "age_col": "AGE_AT_ADT_START",
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
