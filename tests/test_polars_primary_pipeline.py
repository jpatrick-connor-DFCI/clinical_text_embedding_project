"""Regression tests for the v2 Polars-first dataframe boundary."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl


V2_ROOT = Path(__file__).resolve().parents[1] / "v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from pipelines.training.slurm_array_utils import filter_event_rows  # noqa: E402
from shared.polars_utils import filter_finite_rows, finite_or_zero  # noqa: E402
from survival.cox_models.base import scale_model_data  # noqa: E402
from survival.preprocessing import pool_embedding_series_vectorized  # noqa: E402
from survival.preprocessing import map_time_to_event  # noqa: E402
from pipelines.preprocessing.generate_all_non_text_covariates import (  # noqa: E402
    build_cancer_stage_df,
    build_cancer_type_df,
    build_somatic_data_df,
)
from pipelines.preprocessing.build_cohort import _sequencing_dates  # noqa: E402
from pipelines.biomarkers.build_line_matched_cohort import (  # noqa: E402
    build_first_line_new_user_df,
    sequenced_before_landmark,
)
from pipelines.biomarkers.profile_lines import derive_lines_of_therapy  # noqa: E402


class PolarsPrimaryPipelineTests(unittest.TestCase):
    def test_smoke_notebook_defines_subprocess_helpers_before_use(self) -> None:
        notebook_path = V2_ROOT / "notebooks" / "04_smoke_test_training.ipynb"
        notebook = json.loads(notebook_path.read_text())
        code_cells = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            code_cells.append("".join(source) if isinstance(source, list) else source)

        execution_idx = next(
            i for i, source in enumerate(code_cells)
            if "run_results: list[dict]" in source
        )
        preceding_source = "\n".join(code_cells[:execution_idx])
        self.assertIn("def run_subprocess(", preceding_source)
        self.assertIn("def build_full_cohort_cmd(", preceding_source)

    def test_filter_event_rows_rejects_null_nan_infinite_and_nonpositive_times(self) -> None:
        frame = pl.DataFrame(
            {
                "DFCI_MRN": [1, 2, 3, 4, 5, 6, 7],
                "event": [1.0, 0.0, 1.0, 0.0, 1.0, float("nan"), None],
                "tt_event": [10.0, 0.0, -1.0, float("nan"), float("inf"), 5.0, 5.0],
            }
        )

        result = filter_event_rows(frame, "event")

        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result["DFCI_MRN"].to_list(), [1])

    def test_scale_model_data_stays_polars_and_uses_training_statistics(self) -> None:
        train = pl.DataFrame({"age": [10.0, 20.0, 30.0], "binary": [0, 1, 0]})
        test = pl.DataFrame({"age": [40.0], "binary": [1]})

        train_scaled, test_scaled = scale_model_data(train, test, ["age"])

        self.assertIsInstance(train_scaled, pl.DataFrame)
        self.assertIsInstance(test_scaled, pl.DataFrame)
        self.assertEqual(train_scaled["binary"].to_list(), [0, 1, 0])
        np.testing.assert_allclose(train_scaled["age"].mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(test_scaled["age"][0], np.sqrt(6.0), atol=1e-12)

    def test_filter_finite_rows_rejects_every_numeric_missing_value(self) -> None:
        frame = pl.DataFrame(
            {"id": [1, 2, 3, 4, 5], "score": [1.0, None, float("nan"), float("inf"), float("-inf")]}
        )

        result = filter_finite_rows(frame, ["score"])

        self.assertEqual(result["id"].to_list(), [1])

    def test_finite_or_zero_does_not_treat_nan_biomarker_as_positive(self) -> None:
        frame = pl.DataFrame(
            {"marker": [1.0, 0.0, None, float("nan"), float("inf")]}
        )

        marker_positive = frame.select(
            (finite_or_zero("marker") > 0).alias("positive")
        )["positive"].to_list()

        self.assertEqual(marker_positive, [True, False, False, False, False])

    def test_pooled_embedding_frame_does_not_infer_object_feature_columns(self) -> None:
        metadata = pl.DataFrame(
            {
                "DFCI_MRN": [101, 101],
                "NOTE_TYPE": ["Clinician", "Clinician"],
                "NOTE_DATETIME": ["2020-01-01", "2021-01-01"],
                "NOTE_TIME": [-20, -10],
                "EMBEDDING_INDEX": [0, 1],
            }
        )
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result = pool_embedding_series_vectorized(
            metadata,
            embeddings,
            note_types=["Clinician"],
            note_timing_col="NOTE_TIME",
            pool_fx={"Clinician": "mean"},
            year_adj_cols=[],
        )

        self.assertEqual(result.schema["DFCI_MRN"], pl.Int64)
        self.assertTrue(result.schema["CLINICIAN_EMBEDDING_0"].is_float())
        self.assertTrue(result.schema["CLINICIAN_EMBEDDING_1"].is_float())
        filtered = filter_finite_rows(
            result, ["CLINICIAN_EMBEDDING_0", "CLINICIAN_EMBEDDING_1"]
        )
        self.assertEqual(filtered.height, 1)

    def test_events_after_censoring_are_not_cases(self) -> None:
        cohort = pl.DataFrame({"DFCI_MRN": [1, 2], "tt_death": [100.0, 100.0]})
        events = pl.DataFrame({
            "DFCI_MRN": [1, 2], "TIME_TO_EVENT": [80.0, 120.0],
        })

        tt, status = map_time_to_event(
            events, cohort, "DFCI_MRN", "endpoint", "TIME_TO_EVENT"
        )

        self.assertEqual(tt.to_list(), [80.0, 100.0])
        self.assertEqual(status.to_list(), [1, 0])

    def test_compiled_cancer_type_and_note_stage_sources(self) -> None:
        cohort = pl.DataFrame({"DFCI_MRN": list(range(1, 502))})
        cancer_type_source = pl.DataFrame({
            "DFCI_MRN": list(range(1, 502)) + [999],
            "CANCER_GROUP": ["SAFE"] * 500 + ["RARE", "OUTSIDE_COHORT"],
        })
        stage_source = pl.DataFrame({
            "DFCI_MRN": [1, 2],
            "STAGE": ["2", "0"],
        })
        with patch(
            "pipelines.preprocessing.generate_all_non_text_covariates.ps.load_cancer_type",
            return_value=cancer_type_source,
        ), patch(
            "pipelines.preprocessing.generate_all_non_text_covariates.ps.load_cancer_stage",
            return_value=stage_source,
        ):
            cancer_type = build_cancer_type_df(cohort)
            stage = build_cancer_stage_df()

        self.assertEqual(cancer_type.height, 501)
        self.assertEqual(cancer_type.filter(pl.col("DFCI_MRN") == 1)["CANCER_TYPE"].item(), "SAFE")
        self.assertEqual(cancer_type.filter(pl.col("DFCI_MRN") == 501)["CANCER_TYPE"].item(), "OTHER")
        self.assertEqual(stage["DFCI_MRN"].to_list(), [1])
        self.assertEqual(stage["CANCER_STAGE"].to_list(), ["II"])

    def test_somatic_features_require_report_by_anchor(self) -> None:
        cohort = pl.DataFrame({
            "DFCI_MRN": [1], "first_treatment_date": [date(2020, 1, 1)],
        })
        specimens = pl.DataFrame({
            "DFCI_MRN": [1, 1],
            "SAMPLE_ACCESSION_NBR": ["pre", "post"],
            "TEST_TYPE": ["PANEL", "PANEL"],
            "REPORT_DT": [date(2019, 12, 1), date(2020, 2, 1)],
        })
        somatic = pl.DataFrame({
            "DFCI_MRN": [1, 1],
            "SAMPLE_ACCESSION_NBR": ["pre", "post"],
            "TEST_TYPE": ["PANEL", "PANEL"],
            "TP53_SNV": [0, 1],
        })
        target = "pipelines.preprocessing.generate_all_non_text_covariates.ps"
        with patch(f"{target}.load_genomic_specimen", return_value=specimens), patch(
            f"{target}.load_somatic_wide", return_value=somatic
        ):
            result = build_somatic_data_df(cohort)

        self.assertEqual(result["SAMPLE_ACCESSION_NBR"].to_list(), ["pre"])
        self.assertEqual(result["TP53_SNV"].to_list(), [0])

    def test_sequencing_anchor_is_first_report_not_collection_or_order(self) -> None:
        specimens = pl.DataFrame({
            "DFCI_MRN": [1, 1],
            "SAMPLE_COLLECTION_DT": [date(2018, 1, 1), date(2019, 1, 1)],
            "TEST_ORDER_DT": [date(2018, 2, 1), date(2019, 2, 1)],
            "REPORT_DT": [date(2020, 3, 1), date(2020, 1, 1)],
        })
        with patch(
            "pipelines.preprocessing.build_cohort.ps.load_genomic_specimen",
            return_value=specimens,
        ):
            result = _sequencing_dates()

        self.assertEqual(result["sequencing_date"].to_list(), [date(2020, 1, 1)])

    def test_ici_control_status_does_not_depend_on_later_ici(self) -> None:
        # Lines are derived from PROFILE_DATA medications, so the fixture is now
        # a drug-level frame rather than a pre-lined one.
        medications = pl.DataFrame({
            "DFCI_MRN": [1, 1, 2],
            "DRUG": ["carboplatin", "pembrolizumab", "nivolumab"],
            "START_DT": [date(2020, 1, 1),
                         date(2021, 1, 1),
                         date(2020, 1, 1)],
        })
        lines = derive_lines_of_therapy(medications)
        self.assertEqual(lines.sort(["DFCI_MRN", "LINE"])["LINE"].to_list(), [1, 2, 1])

        result = build_first_line_new_user_df(
            lines, {1, 2}, {1: "A", 2: "A"}
        ).sort("DFCI_MRN")

        # Patient 1 later receives ICI, but remains an unexposed line-1
        # initiator because future treatment is not consulted.
        self.assertEqual(result["PX_on_ICI"].to_list(), [0, 1])


    def test_sequencing_eligibility_requires_a_report_at_or_before_the_landmark(self) -> None:
        landmark = pl.DataFrame({
            "DFCI_MRN": [1, 2, 3],
            "treatment_start_date": [date(2020, 1, 1)] * 3,
        })
        specimens = pl.DataFrame({
            "DFCI_MRN": [1, 1, 2, 3],
            "REPORT_DT": [date(2018, 1, 1), date(2019, 6, 1),   # patient 1: two pre-landmark
                          date(2020, 1, 1),                      # patient 2: same day, eligible
                          date(2021, 1, 1)],                     # patient 3: after, ineligible
            "TEST_TYPE": ["ONCOPANEL"] * 4,
        })
        with patch(
            "pipelines.biomarkers.build_line_matched_cohort.ps.load_genomic_specimen",
            return_value=specimens,
        ):
            result = sequenced_before_landmark(landmark).sort("DFCI_MRN")

        self.assertEqual(result["DFCI_MRN"].to_list(), [1, 2])
        # The most recent eligible report is the one reported.
        self.assertEqual(result["sequencing_date"].to_list(), [date(2019, 6, 1), date(2020, 1, 1)])


if __name__ == "__main__":
    unittest.main()
