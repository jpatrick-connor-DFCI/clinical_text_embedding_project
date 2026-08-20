"""Regression tests for the guards that keep an IPTW run from failing silently.

Both guard against the same failure: every model fit funnels through
`filter_finite_rows`, which casts covariates to Float64 with strict=False. A
single non-numeric covariate becomes all-null, drops every row, and makes each
marker fit raise. `_safe_fit` swallowed those, the screen wrote a header-only
CSV, and `compile_IPTW_results` reported it as "0 significant hits".

Skipped where statsmodels/zstandard are absent; they are present on the cluster.
"""

import os
import sys

import polars as pl
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("zstandard")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v2"))

from pipelines.biomarkers.run_IPTW_analysis import (  # noqa: E402
    _run_marker_screen,
    assert_model_covariates_numeric,
)


def _frame():
    return pl.DataFrame({
        "tt_death": [10.0, 20.0, 30.0],
        "death": [1, 0, 1],
        "PX_on_ICI": [1, 0, 1],
        "GENDER": [0, 1, 0],
        "CANCER_TYPE": ["LUNG", "BREAST", "LUNG"],   # raw label kept beside dummies
        "CANCER_TYPE_LUNG": [1, 0, 1],
        "PANEL_VERSION": ["OP_v3", "OP_v3", "OP_v2"],
        "PANEL_VERSION_OP_v3": [1, 1, 0],
    })


def test_raw_cancer_type_label_is_rejected_as_a_covariate():
    with pytest.raises(TypeError, match="CANCER_TYPE"):
        assert_model_covariates_numeric(
            _frame(), ["GENDER", "CANCER_TYPE", "CANCER_TYPE_LUNG"], "spec/pan_cancer")


def test_raw_panel_version_label_is_rejected_as_a_covariate():
    with pytest.raises(TypeError, match="PANEL_VERSION"):
        assert_model_covariates_numeric(
            _frame(), ["GENDER", "PANEL_VERSION"], "spec/pan_cancer")


def test_dummy_only_covariates_pass():
    assert_model_covariates_numeric(
        _frame(),
        ["GENDER", "CANCER_TYPE_LUNG", "PANEL_VERSION_OP_v3", "tt_death", "death", "PX_on_ICI"],
        "spec/pan_cancer",
    )


def test_prefix_filters_exclude_the_raw_label_columns():
    """The filters main() uses to assemble base_vars. A loose `'CANCER_TYPE' in c`
    substring test is what swept the string column in."""
    cols = _frame().columns
    assert [c for c in cols if c.startswith("CANCER_TYPE_")] == ["CANCER_TYPE_LUNG"]
    assert [c for c in cols if c.upper().startswith("PANEL_VERSION_")] == ["PANEL_VERSION_OP_v3"]


def test_screen_raises_when_every_fit_fails():
    def always_fails(df, marker, base_vars, weights_col):
        raise ValueError("no rows left after filter_finite_rows")

    with pytest.raises(RuntimeError, match="all 3 marker fits failed"):
        _run_marker_screen(_frame(), ["m1", "m2", "m3"], [], None,
                           always_fails, 1, label="T2 pan_cancer ATE")


def test_screen_tolerates_partial_failure():
    def fails_one(df, marker, base_vars, weights_col):
        if marker == "m1":
            raise ValueError("singular matrix")
        return {"marker": marker}

    results, failed = _run_marker_screen(_frame(), ["m1", "m2"], [], None,
                                        fails_one, 1, label="partial")
    assert [r["marker"] for r in results] == ["m2"]
    assert failed == [("m1", "singular matrix")]


def test_screen_with_no_markers_does_not_raise():
    def unused(df, marker, base_vars, weights_col):
        raise AssertionError("should not be called")

    assert _run_marker_screen(_frame(), [], [], None, unused, 1, label="none") == ([], [])


# ---------------------------------------------------------------------------
# merge_rare_cancer_types_into_other must return a reference-dropped dummy set.
#
# Upstream build_cancer_type_df uses drop_first=True, so the frame arriving at
# the merge is already reference-dropped. Folding rare types back into
# CANCER_TYPE_OTHER re-adds that column and restores the complete partition,
# which sums to the all-ones vector and makes the Cox partial-likelihood Hessian
# singular -- every fit in the screen then fails with "matrix inversion
# problems". These tests pin the invariant that makes the model identifiable.
# ---------------------------------------------------------------------------

def _cancer_frame(labels, min_total_rows=None):
    """One row per patient, dummies for every label present (complete partition)."""
    import polars as pl
    uniq = sorted(set(labels))
    return pl.DataFrame({
        f"CANCER_TYPE_{u}": [1 if l == u else 0 for l in labels] for u in uniq
    })


def test_merge_rare_drops_a_reference_level():
    from pipelines.biomarkers.run_IPTW_analysis import merge_rare_cancer_types_into_other
    labels = ["LUNG"] * 50 + ["SKIN"] * 40 + ["KIDNEY"] * 35
    out, kept, rare = merge_rare_cancer_types_into_other(_cancer_frame(labels), min_total=30)
    # Three types clear the threshold; one of them must be held out as reference.
    assert len(kept) == 2, kept
    assert "CANCER_TYPE_OTHER" not in kept


def test_merged_dummies_do_not_form_a_complete_partition():
    """The returned columns must not sum to all-ones -- that is the singularity."""
    from pipelines.biomarkers.run_IPTW_analysis import merge_rare_cancer_types_into_other
    labels = ["LUNG"] * 50 + ["SKIN"] * 40 + ["RARE1"] * 5 + ["RARE2"] * 3
    out, kept, rare = merge_rare_cancer_types_into_other(_cancer_frame(labels), min_total=30)
    assert set(rare) == {"CANCER_TYPE_RARE1", "CANCER_TYPE_RARE2"}
    row_sums = out.select(kept).sum_horizontal().to_list()
    # The reference-level patients (the merged-rare OTHER group) are all-zero.
    assert min(row_sums) == 0, "no reference group: dummies form a complete partition"
    assert max(row_sums) == 1, "dummies must remain mutually exclusive"


def test_merge_rare_is_safe_when_every_type_is_rare():
    """Everything collapses into OTHER, leaving no identifiable contrast."""
    from pipelines.biomarkers.run_IPTW_analysis import merge_rare_cancer_types_into_other
    labels = ["A"] * 5 + ["B"] * 4
    out, kept, rare = merge_rare_cancer_types_into_other(_cancer_frame(labels), min_total=30)
    assert kept == []
