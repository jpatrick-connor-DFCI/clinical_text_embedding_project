"""Khorana score vs. the held-out text risk score for venous thromboembolism.

The VTE counterpart of `figures/prep/published_scores.py`, whose evaluation
functions it reuses unchanged. The score is the full Khorana score
(`shared.published_scores.build_khorana`: site, platelets, hemoglobin or ESA,
WBC, BMI) as computed by build_published_scores.py. It is evaluated within its
eligible (solid tumor or lymphoma), lab-observable, complete-case population,
against the `vte` scheme's out-of-fold full-cohort text risk score for the
same event. Patients with a VTE code on or before the anchor are already out
of the `vte` scheme's endpoints (generate_embedding_prediction_datasets.py).

Each run is one (anchor, lab window, event, horizon):
- anchor: treatment (primary) or sequencing;
- lab window: 30 days (primary) or 90 days, the builder's `--lab-window-days`;
- event: `vte` (curated acute PE/DVT ICD-10, primary) or `vte_phecode`
  (PE/DVT phecodes, sensitivity);
- horizon: full follow-up (`horizon_days` null), or 180 days, Khorana's
  validated window, by administrative censoring (events after day 180 become
  censored at 180). The text model is trained on full follow-up either way.

Models, contrasts and statuses are those of published_scores.py: `published`
(Khorana points, unfitted), `text`, and `published+text` (cross-fitted), on
one shared set of comparable pairs blocked by the text model's outer fold.
Bootstrap CIs are computed for the primary runs only (treatment anchor,
30-day window, `vte`, both horizons).

Writes to FIGURE_DATA_DIR the published_scores.py tables, each with `event`
and `horizon_days` columns after `lab_window_days`:
khorana_vte_cindex.csv, khorana_vte_delta.csv, khorana_vte_cox.csv,
khorana_vte_km.csv, khorana_vte_cohort.csv. In khorana_vte_cohort.csv,
`unblocked_published_cindex` is comparable to Khorana validation studies.

Command line: --n-boot (default 1000), --n-jobs.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import as_completed

import polars as pl

from figures.io import save_figure_data
from figures.prep.figure3_combined import DEFAULT_N_BOOT
from figures.prep.parallel import process_pool, resolve_workers
from figures.prep.published_scores import (
    CINDEX_SCHEMA, COHORT_SCHEMA, COX_SCHEMA, DELTA_SCHEMA, KM_SCHEMA, PRIMARY_ANCHOR, PRIMARY_LAB_WINDOW,
    _load_outcomes, _load_text_scores, _missing_outputs, _score_cohort, _score_df_path,
    evaluate_cohort, evaluate_cox, evaluate_km, evaluate_score,
)
from figures.prep.within_cancer import _READ_ERRORS
from shared.published_scores import default_catalog

ANCHORS_TO_RUN = ("treatment", "sequencing")
LAB_WINDOWS_TO_RUN = (30, 90)
EVENTS_TO_RUN = ("vte", "vte_phecode")
HORIZONS_TO_RUN = (None, 180)
PRIMARY_EVENT = "vte"
VTE_SCHEME = "vte"

KHORANA = default_catalog()["khorana"]
SOURCE = "calculated"


def _with_run_keys(schema: dict) -> dict:
    """`schema` with `event` and `horizon_days` inserted after `lab_window_days`."""
    out = {}
    for name, dtype in schema.items():
        out[name] = dtype
        if name == "lab_window_days":
            out["event"] = pl.String
            out["horizon_days"] = pl.Int64
    return out


SCHEMAS = tuple(_with_run_keys(s) for s in (CINDEX_SCHEMA, DELTA_SCHEMA, COX_SCHEMA, KM_SCHEMA, COHORT_SCHEMA))
OUTPUT_NAMES = tuple(f"khorana_vte_{name}.csv" for name in ("cindex", "delta", "cox", "km", "cohort"))


def apply_horizon(outcomes: pl.DataFrame, horizon_days: int | None) -> pl.DataFrame:
    """Administratively censor `time`/`event_flag` at `horizon_days` (None:
    full follow-up)."""
    if horizon_days is None:
        return outcomes
    return outcomes.with_columns(
        pl.when(pl.col("time") > horizon_days).then(0.0).otherwise(pl.col("event_flag")).alias("event_flag"),
        pl.col("time").clip(upper_bound=horizon_days),
    )


def _evaluate_task(task: tuple) -> tuple[pl.DataFrame, ...]:
    anchor, lab_window_days, event, horizon_days, n_boot = task
    keys = {"anchor": anchor, "lab_window_days": lab_window_days, "source": SOURCE}
    try:
        score_df = pl.read_csv(_score_df_path(anchor, lab_window_days), schema_overrides={"DFCI_MRN": pl.String})
        endpoint = (VTE_SCHEME, event)
        text = _load_text_scores(anchor, endpoint)
        outcomes = apply_horizon(_load_outcomes(anchor, endpoint), horizon_days)
        frame = _score_cohort(score_df, text, outcomes, KHORANA)
    except _READ_ERRORS:
        outputs = _missing_outputs(anchor, lab_window_days, KHORANA, SOURCE)
    else:
        cindex, delta = evaluate_score(frame, score=KHORANA, n_boot=n_boot, **keys)
        outputs = (
            cindex, delta,
            evaluate_cox(frame, score=KHORANA, **keys),
            evaluate_km(frame, score=KHORANA, **keys),
            evaluate_cohort(frame, score_df, score=KHORANA, **keys),
        )
    run_keys = (pl.lit(event).alias("event"), pl.lit(horizon_days, dtype=pl.Int64).alias("horizon_days"))
    return tuple(
        out.with_columns(*run_keys).select(list(schema)).cast(schema) for out, schema in zip(outputs, SCHEMAS)
    )


def _tasks(n_boot: int) -> list[tuple]:
    tasks = []
    for anchor in ANCHORS_TO_RUN:
        for lab_window_days in LAB_WINDOWS_TO_RUN:
            for event in EVENTS_TO_RUN:
                primary = (anchor, lab_window_days, event) == (PRIMARY_ANCHOR, PRIMARY_LAB_WINDOW, PRIMARY_EVENT)
                for horizon_days in HORIZONS_TO_RUN:
                    tasks.append((anchor, lab_window_days, event, horizon_days, n_boot if primary else 0))
    return tasks


def prepare_khorana_vte(n_boot: int = DEFAULT_N_BOOT, n_jobs: int | None = None) -> tuple[pl.DataFrame, ...]:
    tasks = _tasks(n_boot)
    with process_pool(min(resolve_workers(n_jobs), max(len(tasks), 1)),
                      initializer=warnings.simplefilter, initargs=("ignore",)) as pool:
        if pool is None:
            outputs = [_evaluate_task(task) for task in tasks]
        else:
            futures = [pool.submit(_evaluate_task, task) for task in tasks]
            for _ in as_completed(futures):
                pass
            outputs = [future.result() for future in futures]
    return tuple(
        pl.concat([out[i] for out in outputs]) if outputs else pl.DataFrame(schema=schema)
        for i, schema in enumerate(SCHEMAS)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.n_boot < 0:
        parser.error("--n-boot must be >= 0")
    for frame, name in zip(prepare_khorana_vte(n_boot=args.n_boot, n_jobs=args.n_jobs), OUTPUT_NAMES):
        save_figure_data(frame, name)


if __name__ == "__main__":
    main()
