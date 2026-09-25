"""Unit tests for pipelines/preprocessing/build_published_scores.py's
windowing, pairing and eligibility logic, using small synthetic frames (no
cluster data). An identity harmonizer stands in for consolidate_dfci_labs."""

import datetime as dt

import polars as pl

from pipelines.preprocessing.build_published_scores import (
    _latest_day_median,
    _paired_latest_day,
    build_eligibility,
)


def _labs_df(rows):
    """rows: list of (mrn, analyte, collect_dt, value)."""
    return pl.DataFrame({
        "DFCI_MRN": [r[0] for r in rows],
        "_analyte_col_name": [r[1] for r in rows],
        "_collect_dt": [r[2] for r in rows],
        "_value": [r[3] for r in rows],
        "_collect_day": [r[2].date() for r in rows],
    }).with_columns(
        pl.col("_collect_dt").cast(pl.Datetime),
    )


def _with_days_before(labs, anchor_dt):
    return labs.with_columns(
        (pl.lit(anchor_dt).cast(pl.Datetime) - pl.col("_collect_dt")).dt.total_days().alias("_days_before_anchor")
    )


class TestLatestDayMedian:
    def test_window_boundaries(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            ("P1", "ldh", dt.datetime(2024, 6, 2), 300.0),   # -1 day: after anchor, excluded
            ("P1", "ldh", dt.datetime(2024, 6, 1), 250.0),   # day 0: included
            ("P1", "ldh", dt.datetime(2024, 5, 1), 200.0),   # 31 days: excluded (window=30)
        ])
        labs = _with_days_before(labs, anchor)
        out = _latest_day_median(labs, "ldh", "anchor", window_days=30, out_name="ldh")
        row = out.filter(pl.col("DFCI_MRN") == "P1")
        assert row.height == 1
        assert row["ldh"].item() == 250.0
        assert row["ldh__days_before_anchor"].item() == 0

    def test_latest_day_wins_over_earlier_day(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            ("P1", "ldh", dt.datetime(2024, 5, 25), 400.0),
            ("P1", "ldh", dt.datetime(2024, 5, 20), 100.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _latest_day_median(labs, "ldh", "anchor", window_days=30, out_name="ldh")
        assert out.filter(pl.col("DFCI_MRN") == "P1")["ldh"].item() == 400.0

    def test_median_of_same_day_values(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            ("P1", "ldh", dt.datetime(2024, 5, 30, 8, 0), 100.0),
            ("P1", "ldh", dt.datetime(2024, 5, 30, 16, 0), 300.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _latest_day_median(labs, "ldh", "anchor", window_days=30, out_name="ldh")
        assert out.filter(pl.col("DFCI_MRN") == "P1")["ldh"].item() == 200.0


class TestPairedLatestDay:
    def test_same_day_pairing_required(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            ("P1", "anc", dt.datetime(2024, 5, 20), 5.0),   # ANC alone on day A
            ("P1", "wbc", dt.datetime(2024, 5, 25), 8.0),   # WBC alone on day B (later)
            ("P1", "anc", dt.datetime(2024, 5, 15), 4.0),   # both present on day C (earliest)
            ("P1", "wbc", dt.datetime(2024, 5, 15), 9.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _paired_latest_day(labs, "anc", "wbc", window_days=30, out_a="dnlr_anc", out_b="dnlr_wbc")
        row = out.filter(pl.col("DFCI_MRN") == "P1")
        assert row.height == 1
        # Must pick day C (both measured), not day A or B individually.
        assert row["dnlr_anc"].item() == 4.0
        assert row["dnlr_wbc"].item() == 9.0

    def test_no_pair_when_never_same_day(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            ("P1", "anc", dt.datetime(2024, 5, 20), 5.0),
            ("P1", "wbc", dt.datetime(2024, 5, 25), 8.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _paired_latest_day(labs, "anc", "wbc", window_days=30, out_a="dnlr_anc", out_b="dnlr_wbc")
        assert out.filter(pl.col("DFCI_MRN") == "P1").height == 0


class TestEligibility:
    def _base_frames(self):
        cohort_df = pl.DataFrame({
            "DFCI_MRN": ["P1", "P2", "P3", "P4"],
            "first_treatment_date": [
                dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1),
                dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1),
            ],
        })
        careg = pl.DataFrame({
            "DFCI_MRN": ["P1", "P2", "P3", "P4"],
            "DIAGNOSIS_DT": [
                dt.datetime(2023, 1, 1),   # P1: before anchor, stage IV -> advanced
                dt.datetime(2024, 7, 1),   # P2: AFTER anchor -> must be ignored for eligibility
                dt.datetime(2023, 1, 1),
                dt.datetime(2023, 1, 1),
            ],
            "_REGISTRY_STAGE": [4, 4, 1, 1],
            "HISTOLOGY_DESC": [
                "Hepatocellular Carcinoma", "Hepatocellular Carcinoma",
                "Small Cell Lung Cancer", "Diffuse Large B-Cell Lymphoma, NOS",
            ],
        })
        cancer_group = pl.DataFrame({
            "DFCI_MRN": ["P1", "P2", "P3", "P4"],
            "CANCER_GROUP": ["LIVER", "LIVER", "LUNG", "AGGR_NHL"],
        })
        met_burden = pl.DataFrame({
            "DFCI_MRN": ["P1", "P2", "P3", "P4"],
            "N_MET_SITES": [1, 0, 1, 0],
        })
        return cohort_df, careg, cancer_group, met_burden

    def test_careg_after_anchor_is_ignored_for_stage_iv_advanced(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p1 = out.filter(pl.col("DFCI_MRN") == "P1")
        p2 = out.filter(pl.col("DFCI_MRN") == "P2")
        assert p1["_advanced"].item() is True   # stage IV diagnosed before anchor
        assert p2["_advanced"].item() is False  # stage IV diagnosed AFTER anchor: not counted

    def test_hcc_eligible_for_albi_meld(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p1 = out.filter(pl.col("DFCI_MRN") == "P1")
        assert p1["albi__eligible"].item() is True
        assert p1["meld__eligible"].item() is True

    def test_sclc_excluded_from_lipi(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p3 = out.filter(pl.col("DFCI_MRN") == "P3")
        assert p3["lipi__eligible"].item() is False  # SCLC histology, excluded

    def test_aggr_nhl_kept_for_ipi(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p4 = out.filter(pl.col("DFCI_MRN") == "P4")
        assert p4["ipi_noecog__eligible"].item() is True
        assert p4["_is_dlbcl"].item() is True


# NOTE: build_lab_features's harmonizer round-trip (to_pandas/from_pandas)
# requires pyarrow, which this local dev environment doesn't have (same
# class of gap as numpy for test_figure3_combined_prep.py). That boundary is
# exercised on the cluster per the plan's handoff notes ("Verify locally
# with synthetic pytest fixtures only"); _latest_day_median, _paired_latest_day
# and build_eligibility above cover the windowing/pairing/eligibility logic
# that doesn't require crossing the pandas boundary.
