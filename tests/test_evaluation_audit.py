"""Regression tests for persisted time-dependent-AUC audit artifacts."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl


V2_ROOT = Path(__file__).resolve().parents[1] / "v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from pipelines.training.slurm_array_utils import write_ipcw_reference_csv  # noqa: E402
from survival.cox_models.base import run_base_CoxPH  # noqa: E402
from survival.cox_models.grid_search import run_grid_CoxPH_parallel  # noqa: E402
from survival.cox_models.heldout import get_nested_heldout_risk_scores_CoxPH  # noqa: E402


def _survival_frame(n: int = 160) -> pl.DataFrame:
    rng = np.random.default_rng(1234)
    x = rng.normal(size=n)
    return pl.DataFrame(
        {
            "DFCI_MRN": np.arange(1000, 1000 + n),
            "x": x,
            "event": np.arange(n) % 2,
            "tstop": rng.uniform(30, 3000, size=n) + 20 * x,
        }
    )


class EvaluationAuditTests(unittest.TestCase):
    def test_penalized_grid_persists_auc_curves_and_ipcw_populations(self) -> None:
        test_df, val_df, _model, ipcw_df = run_grid_CoxPH_parallel(
            _survival_frame(),
            base_cols=[],
            continuous_vars=["x"],
            penalized_cols=["x"],
            l1_ratios=[0.5],
            alphas_to_test=[0.001],
            n_splits=2,
            n_jobs=1,
            max_iter=100,
            return_audit=True,
        )

        eval_times = json.loads(test_df["auc_eval_times"][0])
        self.assertEqual(len(eval_times), 50)
        self.assertEqual(len(json.loads(test_df["auc_curve"][0])), len(eval_times))
        fold_curves = json.loads(val_df["fold_auc_curves"][0])
        self.assertEqual(len(fold_curves), 2)
        self.assertTrue(all(len(curve) == len(eval_times) for curve in fold_curves))
        self.assertEqual(ipcw_df.height, 3)  # two CV folds plus held-out test
        self.assertEqual(set(ipcw_df["eval_data"]), {"cv", "test"})
        self.assertTrue(val_df["mean_c_index"].is_finite().all())
        self.assertEqual(len(json.loads(val_df["fold_c_indices"][0])), 2)

        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "ipcw.csv.gz"
            write_ipcw_reference_csv(ipcw_df, str(fp))
            with gzip.open(fp, "rt", encoding="utf-8") as handle:
                round_trip = pl.read_csv(handle)
            self.assertEqual(round_trip.height, ipcw_df.height)

    def test_base_model_persists_both_metrics_and_auc_audit(self) -> None:
        result_df, ipcw_df = run_base_CoxPH(
            _survival_frame(),
            base_cols=["x"],
            continuous_vars=["x"],
            event_col="event",
            tstop_col="tstop",
            n_splits=2,
            max_iter=100,
            return_audit=True,
        )

        self.assertIn("mean_auc(t)", result_df.columns)
        self.assertIn("mean_c_index", result_df.columns)
        self.assertIn("auc_curve", result_df.columns)
        self.assertIn("fold_auc_curves", result_df.columns)
        self.assertEqual(ipcw_df.height, 3)

    def test_nested_scores_persist_outer_fold_and_fold_specific_hyperparameters(self) -> None:
        scores = get_nested_heldout_risk_scores_CoxPH(
            _survival_frame(),
            base_cols=[],
            continuous_vars=["x"],
            penalized_cols=["x"],
            l1_ratios=[0.5],
            alphas_to_test=[0.001],
            event_col="event",
            tstop_col="tstop",
            n_splits=2,
            n_jobs=1,
            max_iter=100,
        )

        self.assertEqual(scores.height, 160)
        self.assertEqual(set(scores["outer_fold"]), {0, 1})
        self.assertTrue(scores["risk_score"].is_finite().all())
        self.assertEqual(set(scores["selected_l1_ratio"]), {0.5})
        self.assertEqual(set(scores["selected_alpha"]), {0.001})


if __name__ == "__main__":
    unittest.main()
