"""Unit tests for the PROFILE_DATA line-of-therapy derivation.

Synthetic frames only — no cluster data required.
"""

import datetime as dt
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.biomarkers.profile_lines import (  # noqa: E402
    LINE_WINDOW_DAYS,
    derive_lines_of_therapy,
    is_ici_expr,
)

D = dt.date


def _long(rows):
    """Build a minimal unpivot_medications_summary()-shaped frame."""
    return pl.DataFrame(
        rows,
        schema={"DFCI_MRN": pl.Int64, "DRUG": pl.String, "START_DT": pl.Date},
        orient="row",
    )


def test_same_day_combination_is_one_line():
    lines = derive_lines_of_therapy(_long([
        (1, "ipilimumab", D(2020, 1, 1)),
        (1, "nivolumab", D(2020, 1, 1)),
    ]))
    assert lines.height == 1
    assert lines["LINE"].to_list() == [1]
    assert lines["n_drugs"].to_list() == [2]
    assert lines["HAS_ICI"].to_list() == [1]


def test_starts_40_days_apart_are_two_lines():
    lines = derive_lines_of_therapy(_long([
        (1, "carboplatin", D(2020, 1, 1)),
        (1, "docetaxel", D(2020, 2, 10)),
    ]))
    assert lines["LINE"].to_list() == [1, 2]
    assert lines["treatment_start_date"].to_list() == [D(2020, 1, 1), D(2020, 2, 10)]


def test_start_exactly_at_window_boundary_joins_the_same_line():
    lines = derive_lines_of_therapy(_long([
        (1, "carboplatin", D(2020, 1, 1)),
        (1, "pemetrexed", D(2020, 1, 1) + dt.timedelta(days=LINE_WINDOW_DAYS)),
    ]))
    assert lines.height == 1
    # One day later is past the window and opens line 2.
    lines2 = derive_lines_of_therapy(_long([
        (1, "carboplatin", D(2020, 1, 1)),
        (1, "pemetrexed", D(2020, 1, 1) + dt.timedelta(days=LINE_WINDOW_DAYS + 1)),
    ]))
    assert lines2.height == 2


def test_window_is_measured_from_the_line_start_not_the_previous_drug():
    """Three drugs each 20 days apart span 40 days, so the third opens line 2."""
    lines = derive_lines_of_therapy(_long([
        (1, "drug_a", D(2020, 1, 1)),
        (1, "drug_b", D(2020, 1, 21)),
        (1, "drug_c", D(2020, 2, 10)),
    ]))
    assert lines["LINE"].to_list() == [1, 2]
    assert lines["n_drugs"].to_list() == [2, 1]


def test_ici_in_line_2_only_leaves_line_1_unexposed():
    lines = derive_lines_of_therapy(_long([
        (1, "carboplatin", D(2020, 1, 1)),
        (1, "pembrolizumab", D(2020, 6, 1)),
    ]))
    assert lines["LINE"].to_list() == [1, 2]
    assert lines["HAS_ICI"].to_list() == [0, 1]


def test_patient_with_no_medication_rows_is_absent_not_null_filled():
    """unpivot_medications_summary() already drops null-START_DT slots; a patient
    with every slot empty simply has no rows and must not appear in the output."""
    lines = derive_lines_of_therapy(_long([
        (1, "carboplatin", D(2020, 1, 1)),
    ]))
    assert lines["DFCI_MRN"].to_list() == [1]

    empty = derive_lines_of_therapy(_long([]))
    assert empty.height == 0
    assert set(["DFCI_MRN", "LINE", "treatment_start_date", "HAS_ICI"]).issubset(empty.columns)


def test_lines_are_numbered_per_patient():
    lines = derive_lines_of_therapy(_long([
        (2, "nivolumab", D(2021, 3, 1)),
        (1, "carboplatin", D(2020, 1, 1)),
        (1, "docetaxel", D(2020, 6, 1)),
    ])).sort(["DFCI_MRN", "LINE"])
    assert lines["DFCI_MRN"].to_list() == [1, 1, 2]
    assert lines["LINE"].to_list() == [1, 2, 1]


def test_derived_frame_is_unique_on_mrn_line():
    lines = derive_lines_of_therapy(_long([
        (1, "a", D(2020, 1, 1)),
        (1, "b", D(2020, 1, 2)),
        (1, "c", D(2020, 5, 1)),
        (2, "d", D(2020, 1, 1)),
    ]))
    assert lines.select(["DFCI_MRN", "LINE"]).is_unique().all()


@pytest.mark.parametrize("name,expected", [
    ("pembrolizumab", True),
    ("PEMBROLIZUMAB", True),
    ("Nivolumab and Relatlimab-rmbw", True),
    ("atezolizumab", True),
    ("carboplatin", False),
    ("trastuzumab", False),          # a mAb, not a checkpoint inhibitor
    ("ramucirumab", False),
    (None, False),
])
def test_is_ici_expr(name, expected):
    df = pl.DataFrame({"DRUG": [name]}, schema={"DRUG": pl.String})
    assert df.select(is_ici_expr("DRUG"))["DRUG"].to_list() == [expected]


def test_negative_window_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        derive_lines_of_therapy(_long([(1, "a", D(2020, 1, 1))]), window_days=-1)
