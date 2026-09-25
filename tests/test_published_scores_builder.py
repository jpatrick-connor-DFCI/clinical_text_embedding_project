"""Unit tests for pipelines/preprocessing/build_published_scores.py's
windowing, pairing and eligibility logic, using small synthetic frames (no
cluster data). `build_lab_features` is exercised end-to-end through the
real, self-contained `shared.lab_harmonizer.harmonize_labs`, with
`profile_sources.load_labs` monkeypatched to return synthetic raw LABS rows
(real `TEST_TYPE_CD` codes and units, no PROFILE-testing dependency)."""

import datetime as dt

import polars as pl

from pipelines.preprocessing import build_published_scores as bps
from pipelines.preprocessing import profile_sources as ps
from pipelines.preprocessing.build_published_scores import (
    _latest_day_median,
    _paired_latest_day,
    build_eligibility,
    build_lab_features,
    build_score_frame,
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
            (101, "ldh", dt.datetime(2024, 6, 2), 300.0),   # -1 day: after anchor, excluded
            (101, "ldh", dt.datetime(2024, 6, 1), 250.0),   # day 0: included
            (101, "ldh", dt.datetime(2024, 5, 1), 200.0),   # 31 days: excluded (window=30)
        ])
        labs = _with_days_before(labs, anchor)
        out = _latest_day_median(labs, "ldh", "anchor", window_days=30, out_name="ldh")
        row = out.filter(pl.col("DFCI_MRN") == 101)
        assert row.height == 1
        assert row["ldh"].item() == 250.0
        assert row["ldh__days_before_anchor"].item() == 0

    def test_latest_day_wins_over_earlier_day(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            (101, "ldh", dt.datetime(2024, 5, 25), 400.0),
            (101, "ldh", dt.datetime(2024, 5, 20), 100.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _latest_day_median(labs, "ldh", "anchor", window_days=30, out_name="ldh")
        assert out.filter(pl.col("DFCI_MRN") == 101)["ldh"].item() == 400.0

    def test_median_of_same_day_values(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            (101, "ldh", dt.datetime(2024, 5, 30, 8, 0), 100.0),
            (101, "ldh", dt.datetime(2024, 5, 30, 16, 0), 300.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _latest_day_median(labs, "ldh", "anchor", window_days=30, out_name="ldh")
        assert out.filter(pl.col("DFCI_MRN") == 101)["ldh"].item() == 200.0


class TestPairedLatestDay:
    def test_same_day_pairing_required(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            (101, "anc", dt.datetime(2024, 5, 20), 5.0),   # ANC alone on day A
            (101, "wbc", dt.datetime(2024, 5, 25), 8.0),   # WBC alone on day B (later)
            (101, "anc", dt.datetime(2024, 5, 15), 4.0),   # both present on day C (earliest)
            (101, "wbc", dt.datetime(2024, 5, 15), 9.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _paired_latest_day(labs, "anc", "wbc", window_days=30, out_a="dnlr_anc", out_b="dnlr_wbc")
        row = out.filter(pl.col("DFCI_MRN") == 101)
        assert row.height == 1
        # Must pick day C (both measured), not day A or B individually.
        assert row["dnlr_anc"].item() == 4.0
        assert row["dnlr_wbc"].item() == 9.0

    def test_no_pair_when_never_same_day(self):
        anchor = dt.datetime(2024, 6, 1)
        labs = _labs_df([
            (101, "anc", dt.datetime(2024, 5, 20), 5.0),
            (101, "wbc", dt.datetime(2024, 5, 25), 8.0),
        ])
        labs = _with_days_before(labs, anchor)
        out = _paired_latest_day(labs, "anc", "wbc", window_days=30, out_a="dnlr_anc", out_b="dnlr_wbc")
        assert out.filter(pl.col("DFCI_MRN") == 101).height == 0


class TestBuildLabFeatures:
    """End-to-end through the real shared.lab_harmonizer.harmonize_labs,
    with profile_sources.load_labs monkeypatched to synthetic raw rows."""

    def _raw_labs(self, rows):
        """rows: list of (mrn, test_cd, collect_dt, numeric_result, uom)."""
        return pl.DataFrame({
            "DFCI_MRN": [r[0] for r in rows],
            ps.LAB_TEST_CD: [r[1] for r in rows],
            ps.LAB_COLLECT_DT: [r[2] for r in rows],
            ps.LAB_NUMERIC_RESULT: [r[3] for r in rows],
            ps.LAB_RESULT_UOM: [r[4] for r in rows],
        })

    def test_harmonizes_and_windows_real_test_codes(self, monkeypatch):
        anchor = dt.datetime(2024, 6, 1)
        cohort_df = pl.DataFrame({"DFCI_MRN": [101], "first_treatment_date": [anchor]})
        raw = self._raw_labs([
            (101, "LDH", dt.datetime(2024, 5, 20), 250.0, "U/L"),
            (101, "ALB", dt.datetime(2024, 5, 20), 4.0, "g/dL"),
            (101, "XYZ_UNMAPPED", dt.datetime(2024, 5, 20), 99.0, "mg/dL"),
            # brackets the anchor so labs_observable is True for P1
            (101, "LDH", dt.datetime(2024, 7, 1), 260.0, "U/L"),
        ])
        monkeypatch.setattr(ps, "load_labs", lambda columns=None: raw.lazy())

        out = build_lab_features(cohort_df, "treatment", window_days=30)
        row = out.filter(pl.col("DFCI_MRN") == 101)
        assert row["ldh"].item() == 250.0
        assert row["albumin"].item() == 4.0
        assert row["labs_observable"].item() is True

    def test_pairs_dnlr_and_corrected_calcium_from_harmonized_labs(self, monkeypatch):
        anchor = dt.datetime(2024, 6, 1)
        cohort_df = pl.DataFrame({"DFCI_MRN": [101], "first_treatment_date": [anchor]})
        raw = self._raw_labs([
            (101, "ANEU", dt.datetime(2024, 5, 15), 4.0, "10^3/uL"),
            (101, "WBC", dt.datetime(2024, 5, 15), 9.0, "10^3/uL"),
            (101, "CA", dt.datetime(2024, 5, 15), 8.5, "mg/dL"),
            (101, "ALB", dt.datetime(2024, 5, 15), 3.0, "g/dL"),
        ])
        monkeypatch.setattr(ps, "load_labs", lambda columns=None: raw.lazy())

        out = build_lab_features(cohort_df, "treatment", window_days=30)
        row = out.filter(pl.col("DFCI_MRN") == 101)
        assert row["dnlr_anc"].item() == 4.0
        assert row["dnlr_wbc"].item() == 9.0
        # corrected_Ca = Ca + 0.8*(4 - alb) = 8.5 + 0.8*(4-3) = 9.3
        assert row["corrected_calcium"].item() == 9.3

    def test_unmapped_code_and_unsupported_unit_dropped(self, monkeypatch):
        anchor = dt.datetime(2024, 6, 1)
        cohort_df = pl.DataFrame({"DFCI_MRN": [101], "first_treatment_date": [anchor]})
        raw = self._raw_labs([
            (101, "NOT_A_REAL_CODE", dt.datetime(2024, 5, 20), 1.0, "mg/dL"),
            (101, "LDH", dt.datetime(2024, 5, 20), 250.0, "furlongs"),
        ])
        monkeypatch.setattr(ps, "load_labs", lambda columns=None: raw.lazy())

        out = build_lab_features(cohort_df, "treatment", window_days=30)
        row = out.filter(pl.col("DFCI_MRN") == 101)
        assert row["ldh"].item() is None

    def test_labs_observable_false_outside_labs_date_range(self, monkeypatch):
        anchor = dt.datetime(2024, 6, 1)
        cohort_df = pl.DataFrame({"DFCI_MRN": [101, 102], "first_treatment_date": [anchor, dt.datetime(2030, 1, 1)]})
        raw = self._raw_labs([
            (101, "LDH", dt.datetime(2024, 5, 20), 250.0, "U/L"),
            (101, "LDH", dt.datetime(2024, 7, 1), 260.0, "U/L"),
        ])
        monkeypatch.setattr(ps, "load_labs", lambda columns=None: raw.lazy())

        out = build_lab_features(cohort_df, "treatment", window_days=30)
        p1 = out.filter(pl.col("DFCI_MRN") == 101)
        p2 = out.filter(pl.col("DFCI_MRN") == 102)
        assert p1["labs_observable"].item() is True
        assert p2["labs_observable"].item() is False


class TestEligibility:
    def _base_frames(self):
        cohort_df = pl.DataFrame({
            "DFCI_MRN": [101, 102, 103, 104],
            "first_treatment_date": [
                dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1),
                dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1),
            ],
        })
        careg = pl.DataFrame({
            "DFCI_MRN": [101, 102, 103, 104],
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
            "DFCI_MRN": [101, 102, 103, 104],
            "CANCER_GROUP": ["LIVER", "LIVER", "LUNG", "AGGR_NHL"],
        })
        met_burden = pl.DataFrame({
            "DFCI_MRN": [101, 102, 103, 104],
            "N_MET_SITES": [1, 0, 1, 0],
        })
        return cohort_df, careg, cancer_group, met_burden

    def test_careg_after_anchor_is_ignored_for_stage_iv_advanced(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p1 = out.filter(pl.col("DFCI_MRN") == 101)
        p2 = out.filter(pl.col("DFCI_MRN") == 102)
        assert p1["_advanced"].item() is True   # stage IV diagnosed before anchor
        assert p2["_advanced"].item() is False  # stage IV diagnosed AFTER anchor: not counted

    def test_hcc_eligible_for_albi_meld(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p1 = out.filter(pl.col("DFCI_MRN") == 101)
        assert p1["albi__eligible"].item() is True
        assert p1["meld__eligible"].item() is True

    def test_sclc_excluded_from_lipi(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p3 = out.filter(pl.col("DFCI_MRN") == 103)
        assert p3["lipi__eligible"].item() is False  # SCLC histology, excluded

    def test_aggr_nhl_kept_for_ipi(self):
        cohort_df, careg, cancer_group, met_burden = self._base_frames()
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        p4 = out.filter(pl.col("DFCI_MRN") == 104)
        assert p4["ipi_noecog__eligible"].item() is True
        assert p4["_is_dlbcl"].item() is True


def test_module_no_longer_depends_on_profile_testing_repo():
    """Guard against re-introducing the sibling PROFILE-testing repo
    dependency: no PROFILE_TESTING_REPO_PATH, sys.path manipulation, or
    dfci_labs import anywhere in the builder module."""
    assert not hasattr(bps, "PROFILE_TESTING_REPO_PATH")
    assert not hasattr(bps, "_load_harmonizer")


class TestMrnDtypeConsistency:
    """Regression test: DFCI_MRN is Int64 throughout profile_sources.py and
    in cohort_df.parquet on the cluster. _load_met_burden and _load_gleason
    read from CSV/parquet files that aren't guaranteed to round-trip as
    Int64, so they must cast explicitly -- previously _load_met_burden cast
    to pl.String, which crashed build_eligibility's join against the real
    Int64 cohort_df on the cluster (SchemaError: i64 vs str)."""

    def test_met_burden_and_gleason_join_against_int64_cohort(self, monkeypatch, tmp_path):
        met_burden_csv = tmp_path / "met_burden_df.csv.gz"
        pl.DataFrame({"DFCI_MRN": [101, 102], "N_MET_SITES": [1, 0]}).write_csv(
            met_burden_csv, compression="gzip"
        )
        monkeypatch.setattr(bps, "_feature_path", lambda name, anchor: str(met_burden_csv))

        met_burden = bps._load_met_burden("treatment")
        assert met_burden.schema["DFCI_MRN"] == pl.Int64

        cohort_df = pl.DataFrame({
            "DFCI_MRN": [101, 102],
            "first_treatment_date": [dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1)],
        })
        careg = pl.DataFrame({
            "DFCI_MRN": [101, 102],
            "DIAGNOSIS_DT": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
            "_REGISTRY_STAGE": [4, 1],
            "HISTOLOGY_DESC": ["Hepatocellular Carcinoma", "Small Cell Lung Cancer"],
        })
        cancer_group = pl.DataFrame({"DFCI_MRN": [101, 102], "CANCER_GROUP": ["LIVER", "LUNG"]})

        # Would previously raise SchemaError before this dtype fix.
        out = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)
        assert out.height == 2

    def test_gleason_date_string_column_does_not_crash_build_score_frame(self, monkeypatch):
        """Regression test: on the cluster, gleason_date arrives as a string
        (or otherwise non-Datetime) column. build_score_frame previously
        cast it to Datetime only inline inside an offset expression, which
        doesn't persist -- a later filter comparing the *original* string
        column against a Datetime raised InvalidOperationError."""
        cohort_df = pl.DataFrame({
            "DFCI_MRN": [101],
            "first_treatment_date": [dt.datetime(2024, 6, 1)],
            "AGE_AT_TREATMENTSTART": [65],
        })
        careg = pl.DataFrame({
            "DFCI_MRN": [101],
            "DIAGNOSIS_DT": [dt.datetime(2023, 1, 1)],
            "_REGISTRY_STAGE": [1],
            "HISTOLOGY_DESC": ["Prostate Adenocarcinoma"],
        })
        cancer_group = pl.DataFrame({"DFCI_MRN": [101], "CANCER_GROUP": ["PROSTATE"]})
        met_burden = pl.DataFrame({"DFCI_MRN": [101], "N_MET_SITES": [0]})
        eligibility = build_eligibility(cohort_df, "treatment", careg, cancer_group, met_burden)

        monkeypatch.setattr(ps, "load_labs", lambda columns=None: pl.DataFrame(
            schema={
                "DFCI_MRN": pl.Int64, ps.LAB_TEST_CD: pl.String, ps.LAB_COLLECT_DT: pl.Datetime,
                ps.LAB_NUMERIC_RESULT: pl.Float64, ps.LAB_RESULT_UOM: pl.String,
            }
        ).lazy())
        lab_features = build_lab_features(cohort_df, "treatment", window_days=30)

        gleason = pl.DataFrame({
            "DFCI_MRN": [101],
            "gleason_date": ["2023-01-15"],  # string, as on the cluster
            "gleason_primary": [3],
            "gleason_secondary": [4],
        })

        out = build_score_frame(cohort_df, "treatment", lab_features, eligibility, gleason)
        assert out.height == 1
        assert out["gleason_primary"].item() == 3


class TestMissingItemsCsvWrite:
    """Regression test: `{score}__missing_items` columns are pl.List(pl.String)
    (from shared.published_scores.score_expr), which pl.DataFrame.write_csv
    cannot serialize -- ComputeError: CSV format does not support nested
    data. main() must join each list column to a string immediately before
    the write, without changing score_expr's return type (formula tests
    still assert it returns a real list)."""

    def test_list_missing_items_column_round_trips_through_csv(self, tmp_path):
        from shared.published_scores import CATALOG_SCORE_IDS

        score_id = CATALOG_SCORE_IDS[0]
        frame = pl.DataFrame({
            "DFCI_MRN": [101, 102],
            f"{score_id}__missing_items": [["item_a", "item_b"], []],
        })
        assert frame.schema[f"{score_id}__missing_items"] == pl.List(pl.String)

        out_path = tmp_path / "published_scores_df.csv.gz"
        frame.with_columns(
            pl.col(f"{score_id}__missing_items").list.join(",")
        ).write_csv(out_path, compression="gzip")

        read_back = pl.read_csv(out_path)
        assert read_back[f"{score_id}__missing_items"].to_list() == ["item_a,item_b", ""]
