"""VTE code classification (shared/vte.py) and the `vte` scheme's endpoints
(generate_embedding_prediction_datasets._add_vte_events)."""

import polars as pl
import pytest

from pipelines.preprocessing.generate_embedding_prediction_datasets import _add_vte_events
from shared.vte import ACUTE, HISTORY, classify_vte_icd, is_vte_phecode


@pytest.mark.parametrize("code, expected", [
    ("I26.02", ACUTE), ("I26.99", ACUTE), ("I80.10", ACUTE), ("I82.401", ACUTE), ("I82.4Z9", ACUTE),
    ("I82.A11", ACUTE), ("I8240", ACUTE),               # undotted; unambiguous parent of acute codes
    ("I82.501", HISTORY), ("Z86.718", HISTORY), ("Z86.71", HISTORY), ("I27.82", HISTORY),
    ("I82.891", HISTORY),
    ("I82.890", None), ("I82.0", None), ("I80.3", None),  # excluded
    ("I82.89", None),                                     # ambiguous: excluded I82.890 vs history I82.891
    ("I82.21", None),                                     # ambiguous: acute I82.210 vs history I82.211
    ("I26.01", None),                                     # septic/other PE, not listed
    ("I82", None), ("C34.1", None), (None, None), ("", None),
])
def test_classify_vte_icd(code, expected):
    assert classify_vte_icd(code) == expected


@pytest.mark.parametrize("phecode, expected", [
    ("415.1", True), ("415.11", True), ("452.2", True), ("452", False), ("452.0", False), ("415", False),
    (None, False),
])
def test_is_vte_phecode(phecode, expected):
    assert is_vte_phecode(phecode) is expected


def _endpoints(icd_rows, phe_rows):
    """Patients 1-6 with 1000 days of follow-up (patient 5: 100 days)."""
    cohort = pl.DataFrame({"DFCI_MRN": [1, 2, 3, 4, 5, 6], "tt_death": [1000.0] * 4 + [100.0, 1000.0]})
    icd = pl.DataFrame(icd_rows, schema={"DFCI_MRN": pl.Int64, "DIAGNOSIS_ICD10_CD": pl.String,
                                         "TIME_TO_ICD": pl.Float64}, orient="row")
    phe = pl.DataFrame(phe_rows, schema={"DFCI_MRN": pl.Int64, "PHECODE": pl.String,
                                         "TIME_TO_ICD": pl.Float64}, orient="row")
    return _add_vte_events(cohort, icd, phe).sort("DFCI_MRN")


def test_vte_endpoints():
    out = _endpoints(
        [
            (1, "I26.99", 50.0), (1, "I82.401", 20.0),   # 1: first acute code after the anchor
            (2, "I82.401", -10.0), (2, "I26.99", 50.0),  # 2: prior acute VTE
            (3, "Z86.718", 0.0), (3, "I26.99", 50.0),    # 3: prior history-of code at the anchor
            (4, "I82.501", 30.0), (4, "I82.0", 40.0),    # 4: only chronic / excluded codes after
            (5, "I26.99", 200.0),                        # 5: event after censoring
        ],
        [(1, "415.11", 60.0), (6, "452.2", -5.0), (5, "415.1", 200.0)],  # 6: prior VTE by phecode only
    )
    assert out["vte"].to_list() == [1.0, None, None, 0.0, 0.0, None]
    assert out["tt_vte"].to_list() == [20.0, None, None, 1000.0, 100.0, None]
    assert out["vte_phecode"].to_list() == [1.0, None, None, 0.0, 0.0, None]
    assert out["tt_vte_phecode"].to_list() == [60.0, None, None, 1000.0, 100.0, None]
