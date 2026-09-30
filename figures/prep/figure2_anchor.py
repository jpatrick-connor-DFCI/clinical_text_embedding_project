"""Pre-compute inputs for the anchor-sensitivity supplement (sequencing-date time-zero).

Reads only the outputs of `notebooks/2_models/05_sequencing_os_comparison.ipynb`: overall
survival (`death_met:death`) with t=0 at sequencing date, fit on two different populations:

- "full"    full sequencing-anchored cohort, text + baseline covariates vs the baseline model
            (run_full_cohort_event -> full_cohort/death/{text,base}_test.csv)
- "common"  patients with stage, treatment, somatic and germline data, text vs each other
            modality (run_feature_comp_task --modality all -> feature_comps/death/{mod}_test.csv)

The two cohorts differ, so models are comparable only within a cohort.

Writes to FIGURE_DATA_DIR:
- fig2_anchor_sensitivity.csv     cohort, model, n, n_events, cindex, mean_auc, ibs
                                   (one row per cohort x model; model in {base, text} for
                                   "full" and MODALITY_ORDER for "common"; n/n_events are the
                                   cohort the model was fit within -- the prediction frame for
                                   "full", the modality's own held-out risk-score file for
                                   "common" -- while the metrics are on its held-out test split)
"""

from __future__ import annotations

import os

import polars as pl

from config import SURV_PATH
from figures.io import save_figure_data
from pipelines.training.slurm_array_utils import filter_event_rows
from schemes import embedding_file, feature_held_out_dir, full_cohort_event_dir, scheme_results_dir
from shared.palette import MODALITY_ORDER

# The endpoint and anchor the sequencing OS notebook fits.
SCHEME = "death_met"
EVENT = "death"
ANCHOR = "sequencing"

ANCHOR_SENSITIVITY_COLUMNS = ["cohort", "model", "n", "n_events", "cindex", "mean_auc", "ibs"]
_SCHEMA = {
    "cohort": pl.String, "model": pl.String, "n": pl.Int64, "n_events": pl.Int64,
    "cindex": pl.Float64, "mean_auc": pl.Float64, "ibs": pl.Float64,
}


def _test_metrics(fp: str) -> dict | None:
    """C-index, mean AUC(t) and IBS from a runner's one-row *_test.csv, or None if unavailable."""
    try:
        df = pl.read_csv(fp)
    except FileNotFoundError:
        print(f"  missing {fp}")
        return None
    if df.is_empty():
        print(f"  empty {fp}")
        return None
    row = df.row(0, named=True)
    return {"cindex": row["mean_c_index"], "mean_auc": row["mean_auc(t)"], "ibs": row.get("mean_ibs")}


def _outcomes() -> pl.DataFrame:
    """DFCI_MRN, event flag and time for every patient with a valid death endpoint."""
    fp = os.path.join(SURV_PATH, embedding_file(SCHEME, ANCHOR))
    df = pl.read_parquet(fp, columns=["DFCI_MRN", EVENT, f"tt_{EVENT}"])
    return filter_event_rows(df, EVENT)


def _full_cohort_rows(outcomes: pl.DataFrame) -> list[dict]:
    d = full_cohort_event_dir(SCHEME, EVENT, ANCHOR)
    n, n_events = outcomes.height, int(outcomes[EVENT].sum())
    rows = []
    for model in ("base", "text"):
        metrics = _test_metrics(os.path.join(d, f"{model}_test.csv"))
        if metrics is not None:
            rows.append({"cohort": "full", "model": model, "n": n, "n_events": n_events, **metrics})
    return rows


def _common_cohort_rows(outcomes: pl.DataFrame) -> list[dict]:
    d = os.path.join(scheme_results_dir(SCHEME, ANCHOR), "feature_comps", EVENT)
    risk_dir = feature_held_out_dir(SCHEME, EVENT, ANCHOR)
    rows = []
    for mod in MODALITY_ORDER:
        metrics = _test_metrics(os.path.join(d, f"{mod}_test.csv"))
        if metrics is None:
            continue
        # Each modality drops its own NaN rows, so its cohort is the patients it scored.
        n = n_events = None
        risk_fp = os.path.join(risk_dir, f"{mod}_risk_scores.csv")
        if os.path.exists(risk_fp):
            scored = pl.read_csv(risk_fp, columns=["DFCI_MRN"]).unique()
            n = scored.height
            n_events = int(scored.join(outcomes, on="DFCI_MRN")[EVENT].sum())
        rows.append({"cohort": "common", "model": mod, "n": n, "n_events": n_events, **metrics})
    return rows


def main() -> None:
    outcomes = _outcomes()
    rows = _full_cohort_rows(outcomes) + _common_cohort_rows(outcomes)
    sensitivity_df = pl.DataFrame(rows, schema=_SCHEMA).select(ANCHOR_SENSITIVITY_COLUMNS)
    save_figure_data(sensitivity_df, "fig2_anchor_sensitivity.csv")


if __name__ == "__main__":
    main()
