"""Check the per-cancer joint Cox refits for the Figure 3 supplement."""

import gzip

import numpy as np
import polars as pl

from figures.prep import figure3
from figures.prep import within_cancer_joint as prep
from shared.palette import SELECTED_CANCER_TYPES


def _event_frame(n, seed, cancer_offset=0):
    rng = np.random.default_rng(seed)
    text = rng.normal(size=n)
    stage = rng.normal(size=n)
    time = rng.exponential(np.exp(-(text + 0.3 * stage)))
    event = (rng.uniform(size=n) < 0.7).astype(int)
    return pl.DataFrame({
        "DFCI_MRN": np.arange(cancer_offset, cancer_offset + n),
        "text_risk_score": text, "stage_risk_score": stage,
        "death": event, "tt_death": time + 1e-3,
    })


def test_fit_joint_cox_matches_main_columns_and_guards_small_slices():
    frame = _event_frame(200, 0)
    rows = figure3._fit_joint_cox(frame, ["text_risk_score", "stage_risk_score"],
                                  scheme="death_met", event="death")
    betas = pl.DataFrame(rows)
    assert betas.columns == figure3.JOINT_BETA_COLUMNS
    assert set(betas["fit_variant"]) == {"unpenalized", "ridge_0.01"}
    text = betas.filter((pl.col("modality") == "text") & (pl.col("fit_variant") == "unpenalized"))
    assert text["beta"].item() > 0 and text["p_value"].item() < 0.05
    assert figure3._fit_joint_cox(frame.head(10), ["text_risk_score", "stage_risk_score"],
                                  scheme="death_met", event="death") == []
    constant = frame.with_columns(pl.lit(1.0).alias("stage_risk_score"))
    assert figure3._fit_joint_cox(constant, ["text_risk_score", "stage_risk_score"],
                                  scheme="death_met", event="death") == []


def test_refits_each_selected_cancer_separately(monkeypatch):
    breast = _event_frame(200, 1)
    lung = _event_frame(10, 2, cancer_offset=1000)
    other = _event_frame(200, 3, cancer_offset=2000)
    merged = pl.concat([breast, lung, other])
    monkeypatch.setattr(prep, "_joint_event_frames", lambda scheme: iter(
        [("death", merged, ["text_risk_score", "stage_risk_score"])]))
    cancer = pl.DataFrame({
        "DFCI_MRN": [str(i) for i in merged["DFCI_MRN"]],
        "cancer_type": ["BREAST"] * 200 + ["LUNG"] * 10 + ["PROSTATE"] * 200,
    })
    betas, fits = prep.within_cancer_joint_betas("death_met", cancer)
    assert betas.columns == prep.WITHIN_CANCER_JOINT_BETA_COLUMNS
    assert set(betas["cancer_type"]) == {"BREAST"}
    assert betas.filter(pl.col("fit_variant") == "unpenalized")["n"].to_list() == [200, 200]
    # Standardized within the stratum: identical to fitting Breast on its own.
    alone = pl.DataFrame(figure3._fit_joint_cox(
        breast, ["text_risk_score", "stage_risk_score"], scheme="death_met", event="death"))
    assert np.allclose(betas["beta"].to_numpy(), alone["beta"].to_numpy())
    status = dict(zip(fits["cancer_type"], fits["status"]))
    assert fits["cancer_type"].to_list() == list(SELECTED_CANCER_TYPES)
    assert status["BREAST"] == "fitted" and status["LUNG"] == "not_fitted"
    assert status["CUP"] == "no_patients"
    assert "PROSTATE" not in status


def test_cancer_labels_are_normalized_and_restricted(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "FEATURE_PATH", str(tmp_path))
    labels = pl.DataFrame({"DFCI_MRN": ["1", "2", "3"], "CANCER_TYPE": ["Breast", " lung ", "OTHER"]})
    with gzip.open(tmp_path / "cancer_type_df.csv.gz", "wt") as handle:
        handle.write(labels.write_csv())
    result = prep._selected_cancer_labels()
    assert result.to_dicts() == [{"DFCI_MRN": "1", "cancer_type": "BREAST"},
                                 {"DFCI_MRN": "2", "cancer_type": "LUNG"}]
