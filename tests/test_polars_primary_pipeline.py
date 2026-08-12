"""Regression tests for the v2 Polars-first dataframe boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl


V2_ROOT = Path(__file__).resolve().parents[1] / "v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from pipelines.training.slurm_array_utils import filter_event_rows  # noqa: E402
from shared.polars_utils import filter_finite_rows, finite_or_zero  # noqa: E402
from survival.cox_models.base import scale_model_data  # noqa: E402
from survival.preprocessing import pool_embedding_series_vectorized  # noqa: E402


class PolarsPrimaryPipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
