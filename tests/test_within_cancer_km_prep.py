"""Check the per-cancer stage/text-risk KM tables for the Figure 2 supplement."""

import numpy as np
import polars as pl

from figures.prep import within_cancer_km as prep
from shared.palette import SELECTED_CANCER_TYPES


def _pooled(n, offset, seed):
    rng = np.random.default_rng(seed)
    stage = rng.integers(1, 5, size=n)
    return pl.DataFrame({
        "DFCI_MRN": np.arange(offset, offset + n),
        "tt_death": rng.exponential(10, size=n) + 0.1,
        "death": (rng.uniform(size=n) < 0.6).astype(float),
        "text_risk_score": rng.normal(size=n) + offset,  # offset shifts pooled quartiles
        "outer_fold": [0] * n,
        "stage_group": [["I", "II", "III", "IV"][s - 1] for s in stage],
        "stage_ordinal": stage,
        "risk_quartile": ["Q4"] * n,
    })


def test_quartiles_are_within_cancer_and_small_types_are_recorded():
    pooled = pl.concat([_pooled(80, 0, 0), _pooled(10, 1000, 1), _pooled(40, 2000, 2)])
    cancer = pl.DataFrame({
        "DFCI_MRN": [str(i) for i in pooled["DFCI_MRN"]],
        "cancer_type": ["BREAST"] * 80 + ["LUNG"] * 10 + ["PROSTATE"] * 40,
    })
    km, cindex = prep.stage_vs_risk_by_cancer(pooled, cancer)
    assert km.columns == list(prep.KM_SCHEMA)
    assert set(km["cancer_type"]) == {"BREAST"}
    # Recomputed within Breast: four equal groups, not the pooled "Q4" labels.
    assert km["risk_quartile"].value_counts()["count"].to_list() == [20] * 4
    status = {(r["cancer_type"], r["predictor"]): r["status"] for r in cindex.iter_rows(named=True)}
    assert cindex["cancer_type"].unique(maintain_order=True).to_list() == list(SELECTED_CANCER_TYPES)
    assert status[("BREAST", "stage")] == "ok" and status[("BREAST", "text_risk")] == "ok"
    assert status[("LUNG", "stage")] == "below_threshold"
    assert status[("CUP", "text_risk")] == "no_patients"
    breast = cindex.filter(pl.col("cancer_type") == "BREAST")
    assert breast["n"].to_list() == [80, 80] and breast["cindex"].is_between(0, 1).all()
