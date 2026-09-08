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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ---------------------------------------------------------------------------
# assert_base_design_is_identifiable: catch a singular design before fitting.
#
# The all-fits-failed guard catches this too, but only after every marker has
# been fitted -- ~1h of compute on a pan-cancer screen. The base design is
# marker-independent, so it is checked once up front.
# ---------------------------------------------------------------------------

def _design(n=400, seed=0, n_types=4, keep_types=3, extra=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, n_types, n)
    d = {
        "PX_on_ICI": rng.integers(0, 2, n).astype(float),
        "AGE_AT_TREATMENTSTART": rng.normal(60, 10, n),
        "GENDER": rng.integers(0, 2, n).astype(float),
    }
    for i in range(keep_types):
        d[f"CANCER_TYPE_{i}"] = (ct == i).astype(float)
    if extra:
        d.update(extra(d, n, rng))
    return pl.DataFrame(d)


def _base_vars(df):
    return [c for c in df.columns if c != "PX_on_ICI"]


def test_healthy_design_passes():
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _design()
    assert_base_design_is_identifiable(df, _base_vars(df), "spec/pan_cancer")


def test_complete_dummy_partition_is_rejected():
    """The bug that failed all 1492 pan-cancer fits: no reference level."""
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _design(keep_types=4)          # all 4 types -> sums to 1 on every row
    with pytest.raises(ValueError, match="complete partition"):
        assert_base_design_is_identifiable(df, _base_vars(df), "spec/pan_cancer")


def test_constant_column_is_rejected():
    import numpy as np
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _design(extra=lambda d, n, rng: {"PANEL_VERSION_X": np.ones(n)})
    with pytest.raises(ValueError, match="constant column"):
        assert_base_design_is_identifiable(df, _base_vars(df), "spec/pan_cancer")


def test_rank_deficient_design_is_rejected():
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _design(extra=lambda d, n, rng: {"DUP": d["GENDER"].copy()})
    with pytest.raises(ValueError, match="linearly dependent"):
        assert_base_design_is_identifiable(df, _base_vars(df), "spec/pan_cancer")


def test_empty_model_frame_defers_to_the_numeric_guard():
    """An all-null covariate is the other guard's job; this one must not crash."""
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = pl.DataFrame({"PX_on_ICI": [1.0, 0.0], "GENDER": [float("nan")] * 2})
    assert_base_design_is_identifiable(df, ["GENDER"], "spec/empty")


# ---------------------------------------------------------------------------
# `treat_col` is a parameter because the guard is also used on frames where the
# treatment column is constant (a single-arm subset), where including it would
# flag the guard's own scaffolding rather than a real design problem.
# ---------------------------------------------------------------------------

def _single_arm_frame():
    """An ICI-only slice: PX_on_ICI is all-1, and SARCOMA/LINE_2 are all-0."""
    return pl.DataFrame({
        "PX_on_ICI":             [1, 1, 1, 1],
        "GENDER":                [0, 1, 0, 1],
        "AGE_AT_TREATMENTSTART": [60.0, 55.0, 70.0, 65.0],
        "CANCER_TYPE_LUNG":      [1, 1, 0, 0],
        "CANCER_TYPE_SARCOMA":   [0, 0, 0, 0],
        "LINE_2":                [0, 0, 0, 0],
    })


_BASE_VARS = ["GENDER", "AGE_AT_TREATMENTSTART", "CANCER_TYPE_LUNG",
              "CANCER_TYPE_SARCOMA", "LINE_2"]


def test_guard_rejects_constant_columns_on_a_single_arm_frame():
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _single_arm_frame()
    with pytest.raises(ValueError, match="constant column"):
        assert_base_design_is_identifiable(df, _BASE_VARS, "spec/single_arm",
                                           treat_col=None)


def test_guard_passes_once_constant_columns_are_dropped():
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _single_arm_frame()
    kept = [c for c in _BASE_VARS if df[c].n_unique() > 1]
    assert kept == ["GENDER", "AGE_AT_TREATMENTSTART", "CANCER_TYPE_LUNG"]
    assert_base_design_is_identifiable(df, kept, "spec/single_arm", treat_col=None)


def test_treat_col_none_excludes_the_constant_treatment_column():
    """PX_on_ICI is all-1 on a single-arm frame; including it flags scaffolding."""
    from pipelines.biomarkers.run_IPTW_analysis import assert_base_design_is_identifiable
    df = _single_arm_frame()
    kept = [c for c in _BASE_VARS if df[c].n_unique() > 1]
    with pytest.raises(ValueError, match="PX_on_ICI"):
        assert_base_design_is_identifiable(df, kept, "spec/single_arm")
    assert_base_design_is_identifiable(df, kept, "spec/single_arm", treat_col=None)


# ---------------------------------------------------------------------------
# D4 -- four-cell support gate for the marker x ICI interaction
# ---------------------------------------------------------------------------

def _support_frame(n_per_cell=40, events_per_cell=(25, 25, 25, 25), marker="TP53_SNV"):
    """Balanced 2x2 (arm x marker) frame with per-cell event counts.

    Cells are ordered (arm, marker) = (0,0), (0,1), (1,0), (1,1) to match
    `events_per_cell`.
    """
    arm, mk, death = [], [], []
    for (a, m), n_events in zip([(0, 0), (0, 1), (1, 0), (1, 1)], events_per_cell):
        arm += [a] * n_per_cell
        mk += [m] * n_per_cell
        death += [1] * n_events + [0] * (n_per_cell - n_events)
    n = len(arm)
    return pl.DataFrame({
        "tt_death": [float(i % 50 + 1) for i in range(n)],
        "death": death,
        "PX_on_ICI": arm,
        marker: mk,
    })


def test_support_gate_accepts_a_well_populated_two_by_two():
    from pipelines.biomarkers.run_IPTW_analysis import marker_has_within_arm_support
    assert marker_has_within_arm_support(_support_frame(), "TP53_SNV")


def test_support_gate_rejects_too_few_marker_positives_in_one_arm():
    """The head-count gate bites independently of the events floor.

    The ICI/marker+ cell holds MIN_MARKER_POS_PER_ARM - 1 patients who are *all*
    deaths, and `min_events_per_group` is lowered so that cell clears the events
    check outright -- leaving the count check as the only gate that can reject.
    (At the shipped defaults the two floors are equal, so they cannot be
    separated without overriding one.)
    """
    from pipelines.biomarkers.run_IPTW_analysis import (
        MIN_MARKER_POS_PER_ARM, marker_has_within_arm_support,
    )
    n_pos = MIN_MARKER_POS_PER_ARM - 1
    df = _support_frame()
    ici_pos = df.filter((pl.col("PX_on_ICI") == 1) & (pl.col("TP53_SNV") == 1))
    shrunk = ici_pos.head(n_pos).with_columns(
        pl.lit(1, dtype=ici_pos.schema["death"]).alias("death"))
    df = pl.concat([
        df.filter(~((pl.col("PX_on_ICI") == 1) & (pl.col("TP53_SNV") == 1))),
        shrunk,
    ])
    assert not marker_has_within_arm_support(
        df, "TP53_SNV", min_events_per_group=n_pos)
    # ... and the same frame passes once the count floor is what admits it.
    assert marker_has_within_arm_support(
        df, "TP53_SNV", min_pos_per_arm=n_pos, min_events_per_group=n_pos)


def test_support_gate_rejects_too_few_events_among_marker_positives():
    from pipelines.biomarkers.run_IPTW_analysis import marker_has_within_arm_support
    df = _support_frame(events_per_cell=(25, 25, 25, 3))
    assert not marker_has_within_arm_support(df, "TP53_SNV")


def test_support_gate_rejects_too_few_events_among_marker_negatives():
    """The marker- cells are checked too; the interaction contrast needs all four.

    This is the D4 addition -- the old gate looked only at marker+ cells, so a
    marker whose comparison group carried three deaths still reached FDR.
    """
    from pipelines.biomarkers.run_IPTW_analysis import marker_has_within_arm_support
    df = _support_frame(events_per_cell=(25, 25, 3, 25))
    assert not marker_has_within_arm_support(df, "TP53_SNV")


def test_support_thresholds_are_set_for_interaction_tests():
    """Pins the D4 floors: an interaction needs ~4x the events of a main effect."""
    from pipelines.biomarkers.run_IPTW_analysis import (
        MIN_EVENTS_PER_MARKER_GROUP, MIN_MARKER_NEG_PER_ARM, MIN_MARKER_POS_PER_ARM,
    )
    assert MIN_MARKER_POS_PER_ARM >= 20
    assert MIN_MARKER_NEG_PER_ARM >= 20
    assert MIN_EVENTS_PER_MARKER_GROUP >= 20


# ---------------------------------------------------------------------------
# D4 -- extreme interaction HRs are excluded before FDR, not annotated after
# ---------------------------------------------------------------------------

def _t2_row(marker, hr, p):
    return {
        "marker": marker, "HR_markerxICI": hr, "p_markerxICI": p,
        "p_marker_ICI": p, "p_marker_nonICI": p,
        "HR_marker_ICI": hr, "HR_marker_nonICI": 1.0,
    }


def test_extreme_hrs_are_dropped_before_the_fdr_denominator():
    from pipelines.biomarkers.run_IPTW_analysis import (
        HR_EXTREME_THRESHOLD, add_track2_fdr_and_labels,
    )
    df = pl.DataFrame([
        _t2_row("TP53_SNV", 1.4, 0.01),
        _t2_row("KRAS_SNV", 0.7, 0.02),
        _t2_row("EGFR_SNV", HR_EXTREME_THRESHOLD * 10, 0.001),      # separation
        _t2_row("BRAF_SNV", 1.0 / (HR_EXTREME_THRESHOLD * 10), 0.001),
    ])
    out = add_track2_fdr_and_labels(df)
    assert sorted(out["marker"].to_list()) == ["KRAS_SNV", "TP53_SNV"]
    # Every surviving row is non-extreme by construction.
    assert not any(out["extreme_hr_flag"].to_list())
    # FDR was computed over 2 markers, not 4: BH on the smaller p of two.
    assert out.filter(pl.col("marker") == "TP53_SNV")["FDR_markerxICI"].item() == pytest.approx(0.02)


def test_non_finite_interaction_hrs_are_also_excluded():
    from pipelines.biomarkers.run_IPTW_analysis import add_track2_fdr_and_labels
    df = pl.DataFrame([
        _t2_row("TP53_SNV", 1.4, 0.01),
        _t2_row("KRAS_SNV", float("inf"), 0.001),
        _t2_row("EGFR_SNV", float("nan"), 0.001),
    ])
    out = add_track2_fdr_and_labels(df)
    assert out["marker"].to_list() == ["TP53_SNV"]


def test_all_extreme_returns_an_empty_frame_with_the_full_schema():
    """An all-separation spec must not crash the FDR step."""
    from pipelines.biomarkers.run_IPTW_analysis import (
        HR_EXTREME_THRESHOLD, add_track2_fdr_and_labels,
    )
    df = pl.DataFrame([_t2_row("TP53_SNV", HR_EXTREME_THRESHOLD * 10, 0.001)])
    out = add_track2_fdr_and_labels(df)
    assert out.is_empty()
    for col in ("mutation_type", "classifier", "extreme_hr_flag"):
        assert col in out.columns


# ---------------------------------------------------------------------------
# D6 -- unpenalized inferential fit, penalized only as a convergence fallback
# ---------------------------------------------------------------------------

def test_fit_reports_convergence_on_a_well_behaved_frame():
    from lifelines import CoxPHFitter
    from pipelines.biomarkers.run_IPTW_analysis import fit_cph_log_warnings
    import numpy as np

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    df_fit = pl.DataFrame({
        "tt_death": rng.exponential(scale=np.exp(-0.3 * x)) + 0.1,
        "death": rng.integers(0, 2, n),
        "x": x,
    }).to_pandas()
    _, converged = fit_cph_log_warnings(
        CoxPHFitter(), df_fit, "tt_death", "death", robust=False,
        marker_name="x", return_converged=True)
    assert converged


def test_fit_reports_non_convergence_under_separation():
    """A perfectly separating covariate warns rather than raising.

    This is why D6's fallback keys on `converged` and not on an exception: an
    exception-only check would have silently kept the unconverged unpenalized fit.
    """
    from lifelines import CoxPHFitter
    from pipelines.biomarkers.run_IPTW_analysis import fit_cph_log_warnings

    n = 60
    # x == death: the covariate perfectly predicts the event.
    df_fit = pl.DataFrame({
        "tt_death": [float(i + 1) for i in range(n)],
        "death": [1] * (n // 2) + [0] * (n // 2),
        "x": [1.0] * (n // 2) + [0.0] * (n // 2),
    }).to_pandas()
    _, converged = fit_cph_log_warnings(
        CoxPHFitter(), df_fit, "tt_death", "death", robust=False,
        marker_name="x", return_converged=True)
    assert not converged


def _track2_frame(n=400, separating=False, seed=0):
    """Cohort frame for `_fit_track2_marker`: one covariate, one binary marker."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, n)
    marker = rng.integers(0, 2, n)
    if separating:
        # Every marker+ ICI patient dies, none of them are censored: the
        # interaction term is perfectly separating.
        death = ((arm == 1) & (marker == 1)).astype(int)
        tt = np.where(death == 1, rng.uniform(0.5, 2.0, n), rng.uniform(5.0, 20.0, n))
    else:
        tt = rng.exponential(scale=np.exp(-0.2 * arm - 0.2 * marker)) + 0.1
        death = rng.integers(0, 2, n)
    return pl.DataFrame({
        "tt_death": tt.astype(float),
        "death": death.astype(int),
        "PX_on_ICI": arm.astype(int),
        "AGE": rng.normal(60, 10, n),
        "TP53_SNV": marker.astype(int),
    })


def test_track2_uses_an_unpenalized_fit_when_the_model_converges():
    """D6: p-values that feed FDR must come from an unpenalized fit by default."""
    from pipelines.biomarkers.run_IPTW_analysis import _fit_track2_marker
    res = _fit_track2_marker(_track2_frame(), "TP53_SNV", ["AGE"], None)
    assert res["penalized_fit"] is False
    assert res["p_markerxICI"] == pytest.approx(res["p_markerxICI"])  # finite, not NaN
    assert 0.0 <= res["p_markerxICI"] <= 1.0


def test_track2_falls_back_to_a_penalized_fit_under_separation():
    """Near-separation surfaces as a warning, not an exception -- the fallback
    keys on convergence so an unconverged unpenalized estimate is never kept."""
    from pipelines.biomarkers.run_IPTW_analysis import _fit_track2_marker
    res = _fit_track2_marker(
        _track2_frame(separating=True), "TP53_SNV", ["AGE"], None)
    assert res["penalized_fit"] is True


# ---------------------------------------------------------------------------
# D2 -- balance diagnostics carry the covariate family through the long form
# ---------------------------------------------------------------------------

def _write_balance_parquet(tmp_path, keys):
    from pipelines.biomarkers.run_IPTW_analysis import _melt_diagnostic
    wide = pl.DataFrame({
        "covariate": keys,
        "smd_unweighted": [0.30, 0.25][: len(keys)],
        "smd_weighted": [0.04, 0.13][: len(keys)],
    })
    path = str(tmp_path / "LUNG_diagnostics.parquet")
    _melt_diagnostic(wide, "balance_ATE", key_col="covariate").write_parquet(path)
    return path


def test_balance_section_round_trips_the_family_prefixed_key(tmp_path):
    from pipelines.biomarkers.run_IPTW_analysis import read_diagnostic_section
    path = _write_balance_parquet(tmp_path, ["structured|AGE", "embedding|emb_7"])
    wide = read_diagnostic_section(path, "balance_ATE")
    assert "key" not in wide.columns
    got = dict(zip(wide["covariate"].to_list(), wide["covariate_family"].to_list()))
    assert got == {"AGE": "structured", "emb_7": "embedding"}
    # The embedding dimension is the one that trips the D5 gate; without D2 it
    # was never measured at all.
    emb = wide.filter(pl.col("covariate") == "emb_7")
    assert emb["smd_weighted"].item() == pytest.approx(0.13)


def test_legacy_unprefixed_balance_files_still_read(tmp_path):
    """Diagnostics written before D2 have no '|' in the key; keep reading them."""
    from pipelines.biomarkers.run_IPTW_analysis import read_diagnostic_section
    path = _write_balance_parquet(tmp_path, ["AGE", "GENDER"])
    wide = read_diagnostic_section(path, "balance_ATE")
    assert "key" in wide.columns
    assert "covariate_family" not in wide.columns
    assert sorted(wide["key"].to_list()) == ["AGE", "GENDER"]


def test_missing_section_returns_an_empty_frame(tmp_path):
    from pipelines.biomarkers.run_IPTW_analysis import read_diagnostic_section
    path = _write_balance_parquet(tmp_path, ["structured|AGE"])
    assert read_diagnostic_section(path, "cohort").is_empty()


# ---------------------------------------------------------------------------
# D5 -- analyzability gate thresholds
# ---------------------------------------------------------------------------

def test_analyzability_thresholds_are_preregistered():
    from pipelines.biomarkers.run_IPTW_analysis import (
        MAX_SMD_FOR_ANALYSIS, MIN_ESS_FRACTION,
    )
    assert MAX_SMD_FOR_ANALYSIS == pytest.approx(0.1)
    assert 0.0 < MIN_ESS_FRACTION < 1.0
