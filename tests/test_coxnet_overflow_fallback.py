"""Final Coxnet fits fall back down the CV ranking when sksurv's weights overflow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sksurv.linear_model import CoxnetSurvivalAnalysis  # noqa: E402

from survival.cox_models import heldout  # noqa: E402
from survival.cox_models.grid_search import run_grid_CoxPH_parallel  # noqa: E402

_OVERFLOW_MSG = "Numerical error, because weights are too large. Consider increasing alpha."
_real_fit = CoxnetSurvivalAnalysis.fit


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


def _grid(**kwargs):
    params = dict(
        base_cols=[], continuous_vars=["x"], penalized_cols=["x"],
        l1_ratios=[0.5], alphas_to_test=[0.1, 0.01, 0.001],
        n_splits=2, n_jobs=1, max_iter=100,
    )
    params.update(kwargs)
    return run_grid_CoxPH_parallel(_survival_frame(), **params)


class GridFinalFitFallbackTests(unittest.TestCase):
    def _patched_final_fit(self, overflow_calls: int):
        """Overflow the first ``overflow_calls`` final (baseline-model) fits; CV fits run normally."""
        tried = []

        def fit(model, X, y):
            if model.fit_baseline_model:
                tried.append(float(model.alphas[0]))
                if len(tried) <= overflow_calls:
                    raise ArithmeticError(_OVERFLOW_MSG)
            return _real_fit(model, X, y)

        return mock.patch.object(CoxnetSurvivalAnalysis, "fit", fit), tried

    def test_overflowing_winner_falls_back_to_next_cv_candidate(self) -> None:
        patch, tried = self._patched_final_fit(overflow_calls=1)
        with patch:
            test_df, val_df, model = _grid()

        ranked = val_df.filter(pl.col("mean_auc(t)").is_finite()).sort("mean_auc(t)", descending=True)
        self.assertIsNotNone(model)
        self.assertEqual(tried[:2], ranked["alpha"].to_list()[:2])
        self.assertEqual(test_df["selected_alpha"][0], ranked["alpha"][1])
        self.assertEqual(test_df["selected_cv_mean_auc(t)"][0], ranked["mean_auc(t)"][1])
        self.assertTrue(np.isfinite(test_df["mean_auc(t)"][0]))

    def test_every_candidate_overflowing_yields_no_model(self) -> None:
        patch, tried = self._patched_final_fit(overflow_calls=10**6)
        with patch:
            test_df, _val_df, model = _grid()

        self.assertIsNone(model)
        self.assertEqual(len(tried), 3)
        self.assertTrue(np.isnan(test_df["mean_auc(t)"][0]))
        self.assertTrue(np.isnan(test_df["selected_alpha"][0]))

    def test_low_alpha_refit_overflow_keeps_primary_model(self) -> None:
        def fit(model, X, y):
            if model.fit_baseline_model and model.alphas[0] < 0.005:
                raise ArithmeticError(_OVERFLOW_MSG)
            return _real_fit(model, X, y)

        with mock.patch.object(CoxnetSurvivalAnalysis, "fit", fit):
            test_df, val_df, model = _grid(alphas_to_test=[0.01], adaptive_low_alphas=[0.001])

        self.assertEqual(set(val_df["alpha"]), {0.01, 0.001})
        self.assertIsNotNone(model)
        self.assertEqual(test_df["selected_alpha"][0], 0.01)


class NestedHeldoutFallbackTests(unittest.TestCase):
    def _run(self, fake):
        with mock.patch.object(heldout, "fit_predict_external_CoxPH", fake):
            return heldout.get_nested_heldout_risk_scores_CoxPH(
                _survival_frame(), base_cols=[], continuous_vars=["x"], penalized_cols=["x"],
                l1_ratios=[0.5], alphas_to_test=[0.1, 0.01, 0.001],
                event_col="event", tstop_col="tstop", n_splits=2, n_jobs=1, max_iter=100,
            )

    def test_overflow_in_outer_fold_uses_next_inner_candidate(self) -> None:
        real = heldout.fit_predict_external_CoxPH
        tried = []

        def fake(*args, **kwargs):
            tried.append(kwargs["alpha"])
            if len(tried) % 2 == 1:  # first attempt in each outer fold overflows
                raise OverflowError(_OVERFLOW_MSG)
            return real(*args, **kwargs)

        scores = self._run(fake)

        self.assertEqual(len(tried), 4)
        self.assertTrue(scores["risk_score"].is_finite().all())
        per_fold = scores.group_by("outer_fold").agg(pl.col("selected_alpha").first()).sort("outer_fold")
        self.assertEqual(per_fold["selected_alpha"].to_list(), [tried[1], tried[3]])

    def test_non_overflow_errors_still_propagate(self) -> None:
        def fake(*args, **kwargs):
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            self._run(fake)

    def test_every_candidate_overflowing_raises(self) -> None:
        def fake(*args, **kwargs):
            raise ArithmeticError(_OVERFLOW_MSG)

        with self.assertRaisesRegex(RuntimeError, "every inner-CV candidate overflowed"):
            self._run(fake)


if __name__ == "__main__":
    unittest.main()
