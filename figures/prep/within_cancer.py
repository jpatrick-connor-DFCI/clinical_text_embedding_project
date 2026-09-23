"""Prepare within-cancer performance supplements from existing held-out scores.

These are cancer-stratified evaluations of pooled pan-cancer models, not models
refitted within each cancer. Figure 2 uses full-cohort text/base scores; Figure 3
uses text and comparator scores from the feature-comparison cohort. Every
comparison uses the same patients, with valid outcomes and both predictions.

Harrell's C is calculated only among patients in the same cancer and the same
(text outer fold, comparator outer fold) block. Each score then comes from one
fixed fitted model within a block, even when the two modalities' fold assignments
differ. Concordant pairs plus half of tied-risk pairs are summed over blocks and
divided by their total comparable pairs. This is a comparable-pair-weighted
out-of-fold estimand; cross-fold risk-score scales are never compared. Fold IDs
are required, including for legacy files. No models are trained here.

Outputs (in FIGURE_DATA_DIR):
  fig2_within_cancer_cindex.csv, fig3_within_cancer_cindex.csv:
    scheme,event,cancer_type,comparator,text_cindex,comparator_cindex,
    delta_cindex,n_patients,n_events,n_comparable_pairs,n_fold_blocks,
    min_patients_required,min_events_required,status
  within_cancer_audit.csv:
    comparison,scheme,event,comparator,status,detail,n_rows
  fig2_within_cancer_event_counts.csv, fig3_within_cancer_event_counts.csv:
    scheme,event,cancer_type,n_patients,n_events,n_non_events,
    min_patients_required,min_events_required,eligible,status
  fig3_within_cancer_modality_cindex.csv:
    scheme,event,cancer_type,modality,cindex,n_patients,n_events,
    n_comparable_pairs,n_fold_blocks,min_patients_required,min_events_required,status

The modality table ranks all modalities on one footing: per endpoint, patients
with every modality's score (text included) are evaluated together, and C is
computed on one shared set of comparable pairs within joint blocks of all the
modalities' outer folds. Endpoints missing any modality's scores are audited
(incomplete_modalities) and omitted from that table only.

Event counts precede score loading, including endpoints without prediction files.
The full source cohort is the scheme's embedding/outcome patients with a cancer
label. The modality source cohort additionally intersects the somatic, germline,
stage and treatment feature-file patient sets, matching training. These counts
are upper bounds before model-specific covariate and score exclusions. Known
cancer strata with no valid observations have zero counts. Ineligible endpoints
skip score loading; matched strata are checked again before concordance work.
Sparse matched strata have null pair/block counts because those are not computed.
Missing/malformed inputs and exclusions are recorded only in the audit CSV.
The run displays one progress bar for all full-cohort events and one for all
modality-cohort events, with Python warnings suppressed for the preparation run.
Endpoints are evaluated in parallel worker processes (--n-jobs; see
figures.prep.parallel); results are assembled in task order, so the outputs
match a serial run exactly.
Only status == 'ok' rows should be plotted. n_fold_blocks counts blocks with
at least one comparable pair; n_patients and n_events describe the full matched
valid cohort, including patients in blocks with no comparable pairs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from functools import reduce
from pathlib import Path
import warnings

import numpy as np
import polars as pl
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored
from tqdm.auto import tqdm

from config import FEATURE_PATH, FIGURE_DATA_DIR, SURV_PATH
from figures.prep.parallel import process_pool, resolve_workers
from schemes import embedding_file, scheme_results_dir
from shared.palette import MODALITY_ORDER

SCHEMES = ("death_met", "icd3_post", "icd4_post", "phecode_post")
# Same membership files as training's _get_common_feature_mrns. Metastatic
# burden is left-joined/zero-filled in training and does not restrict membership.
MODALITY_COHORT_FILES = (
    "complete_somatic_data_df.csv.gz", "complete_germline_data_df.csv.gz",
    "cancer_stage_df.csv.gz", "categorical_treatment_data_by_line.csv.gz",
)
RESULT_SCHEMA = {
    "scheme": pl.String,
    "event": pl.String,
    "cancer_type": pl.String,
    "comparator": pl.String,
    "text_cindex": pl.Float64,
    "comparator_cindex": pl.Float64,
    "delta_cindex": pl.Float64,
    "n_patients": pl.Int64,
    "n_events": pl.Int64,
    "n_comparable_pairs": pl.Int64,
    "n_fold_blocks": pl.Int64,
    "min_patients_required": pl.Int64,
    "min_events_required": pl.Int64,
    "status": pl.String,
}
AUDIT_SCHEMA = {
    "comparison": pl.String,
    "scheme": pl.String,
    "event": pl.String,
    "comparator": pl.String,
    "status": pl.String,
    "detail": pl.String,
    "n_rows": pl.Int64,
}
MODALITY_RESULT_SCHEMA = {
    "scheme": pl.String,
    "event": pl.String,
    "cancer_type": pl.String,
    "modality": pl.String,
    "cindex": pl.Float64,
    "n_patients": pl.Int64,
    "n_events": pl.Int64,
    "n_comparable_pairs": pl.Int64,
    "n_fold_blocks": pl.Int64,
    "min_patients_required": pl.Int64,
    "min_events_required": pl.Int64,
    "status": pl.String,
}
COUNT_SCHEMA = {
    "scheme": pl.String,
    "event": pl.String,
    "cancer_type": pl.String,
    "n_patients": pl.Int64,
    "n_events": pl.Int64,
    "n_non_events": pl.Int64,
    "min_patients_required": pl.Int64,
    "min_events_required": pl.Int64,
    "eligible": pl.Boolean,
    "status": pl.String,
}
_READ_ERRORS = (OSError, ValueError, pl.exceptions.PolarsError)


def _audit(
    rows: list[dict],
    comparison: str,
    scheme: str,
    event: str,
    comparator: str,
    status: str,
    detail: str,
    n_rows: int = 0,
) -> None:
    rows.append({
        "comparison": comparison, "scheme": scheme, "event": event,
        "comparator": comparator, "status": status, "detail": detail,
        "n_rows": n_rows,
    })


def _validate_ids(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    """Reject ambiguous joins rather than inflating patient/pair counts."""
    if "DFCI_MRN" not in frame.columns:
        raise ValueError(f"{source}: missing DFCI_MRN")
    frame = frame.with_columns(pl.col("DFCI_MRN").cast(pl.String).str.strip_chars())
    ids = frame["DFCI_MRN"]
    if ids.null_count() or (ids == "").any():
        raise ValueError(f"{source}: null or blank patient IDs")
    if ids.n_unique() != frame.height:
        raise ValueError(f"{source}: duplicate patient IDs")
    return frame


def _load_cancer_types(path: Path) -> pl.DataFrame:
    cancer = pl.read_csv(
        path,
        schema_overrides={"DFCI_MRN": pl.String, "CANCER_TYPE": pl.String},
    )
    columns = ["DFCI_MRN", "cancer_type"]
    # Match filter_event_rows' brainM exclusion exactly; do not guess the
    # drop-first category when a legacy file lacks this dummy.
    if "CANCER_TYPE_BRAIN" in cancer.columns:
        brain = pl.col("CANCER_TYPE_BRAIN").cast(pl.Float64, strict=False)
        cancer = cancer.with_columns(
            (brain.is_finite() & (brain != 0)).fill_null(False).alias("_primary_brain")
        )
        columns.append("_primary_brain")
    return _validate_ids(cancer, str(path)).with_columns(
        pl.col("CANCER_TYPE").str.strip_chars().alias("cancer_type")
    ).select(columns)


def _load_modality_cohort_ids() -> pl.DataFrame:
    """Read membership only, without loading feature values or embeddings."""
    common = None
    for filename in MODALITY_COHORT_FILES:
        path = Path(FEATURE_PATH) / filename
        ids = pl.read_csv(path, columns=["DFCI_MRN"], schema_overrides={"DFCI_MRN": pl.String})
        ids = ids.with_columns(pl.col("DFCI_MRN").str.strip_chars()).unique()
        ids = _validate_ids(ids, str(path))
        common = ids if common is None else common.join(ids, on="DFCI_MRN", how="inner")
    return common


def _event_cohort(cohort: pl.DataFrame, event: str) -> pl.DataFrame:
    if event == "brainM" and "_primary_brain" in cohort.columns:
        return cohort.filter(~pl.col("_primary_brain"))
    return cohort


def _pre_evaluation_counts(
    outcomes: pl.DataFrame, cohort: pl.DataFrame, *, scheme: str, event: str,
    min_patients: int, min_events: int, cancer_types: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Count source-cohort observations before any score/fold availability filter."""
    # Preserve source-cohort strata even if this endpoint has zero valid rows
    # (including a cancer excluded by a training endpoint rule).
    cancers = (cohort.select("cancer_type") if cancer_types is None else cancer_types).unique()
    valid = outcomes.join(
        _event_cohort(cohort, event).select("DFCI_MRN", "cancer_type"),
        on="DFCI_MRN", how="inner", validate="1:1",
    ).filter(
        pl.col("time").is_finite() & (pl.col("time") > 0)
        & pl.col("event_flag").is_in([0.0, 1.0])
    )
    observed = valid.group_by("cancer_type").agg(
        pl.len().alias("n_patients"),
        pl.col("event_flag").sum().cast(pl.Int64).alias("n_events"),
    )
    counts = cancers.join(observed, on="cancer_type", how="left").with_columns(
        pl.col("n_patients", "n_events").fill_null(0),
        pl.lit(scheme).alias("scheme"), pl.lit(event).alias("event"),
        pl.lit(min_patients).alias("min_patients_required"),
        pl.lit(min_events).alias("min_events_required"),
    ).with_columns(
        (pl.col("n_patients") - pl.col("n_events")).alias("n_non_events"),
        ((pl.col("n_patients") >= min_patients) & (pl.col("n_events") >= min_events)).alias("eligible"),
        pl.when(pl.col("n_patients") < min_patients).then(pl.lit("too_few_patients"))
        .when(pl.col("n_events") < min_events).then(pl.lit("too_few_events"))
        .otherwise(pl.lit("ok")).alias("status"),
    )
    return counts.select(list(COUNT_SCHEMA)).cast(COUNT_SCHEMA).sort("cancer_type")


def _load_scores(path: Path, modality: str, role: str) -> pl.DataFrame:
    scores = _validate_ids(
        pl.read_csv(path, schema_overrides={"DFCI_MRN": pl.String}), str(path)
    )
    if "outer_fold" not in scores.columns:
        raise ValueError(f"{path}: missing outer_fold; regenerate legacy risk scores")
    score = f"{modality}_risk_score"
    if score not in scores.columns:
        if "risk_score" not in scores.columns:
            raise ValueError(f"{path}: missing {score} (or risk_score)")
        score = "risk_score"
    return scores.select(
        "DFCI_MRN",
        pl.col(score).cast(pl.Float64, strict=False).alias(f"{role}_score"),
        pl.col("outer_fold").cast(pl.Float64, strict=False).alias(f"{role}_fold"),
    )


def _read_outcomes(source: pl.LazyFrame, event: str) -> pl.DataFrame:
    """Push the endpoint-only projection into the parquet reader."""
    frame = source.select(
        "DFCI_MRN",
        pl.col(event).cast(pl.Float64, strict=False).alias("event_flag"),
        pl.col(f"tt_{event}").cast(pl.Float64, strict=False).alias("time"),
    ).collect()
    return _validate_ids(frame, f"survival labels for {event}")


def _valid_fold(column: str) -> pl.Expr:
    value = pl.col(column)
    return value.is_finite() & (value >= 0) & (value == value.floor())


def _matched_patients(
    text: pl.DataFrame,
    comparator: pl.DataFrame,
    outcomes: pl.DataFrame,
    cancer: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Match by patient and return counts of mutually exclusive row exclusions."""
    paired = text.join(comparator, on="DFCI_MRN", how="inner", validate="1:1")
    joined = paired.join(outcomes, on="DFCI_MRN", how="inner", validate="1:1")
    joined = joined.join(cancer, on="DFCI_MRN", how="left", validate="1:1")
    exclusions = {
        "unmatched_text_patients": text.height - paired.height,
        "unmatched_comparator_patients": comparator.height - paired.height,
        "missing_outcomes": paired.height - joined.height,
    }
    checks = (
        ("invalid_outcomes", pl.col("time").is_finite() & (pl.col("time") > 0)
         & pl.col("event_flag").is_in([0.0, 1.0])),
        ("invalid_scores", pl.col("text_score").is_finite()
         & pl.col("comparator_score").is_finite()),
        ("invalid_fold_ids", _valid_fold("text_fold") & _valid_fold("comparator_fold")),
        ("missing_cancer_type", pl.col("cancer_type").is_not_null()
         & (pl.col("cancer_type") != "")),
    )
    for reason, valid in checks:
        retained = joined.filter(valid.fill_null(False))
        exclusions[reason] = joined.height - retained.height
        joined = retained
    return joined.with_columns(
        pl.col("text_fold").cast(pl.Int64),
        pl.col("comparator_fold").cast(pl.Int64),
    ), exclusions


def _block_concordance(block: pl.DataFrame, score_cols: list[str]) -> tuple[list[float], int]:
    """Return one concordance numerator per score on one common set of comparable pairs."""
    zeros = [0.0] * len(score_cols)
    if block.height < 2 or block["event_flag"].sum() == 0:
        return zeros, 0
    event = block["event_flag"].cast(pl.Boolean).to_numpy()
    times = block["time"].to_numpy()
    numerators = []
    n_pairs = None
    try:
        for column in score_cols:
            # Some sksurv releases return NaN plus zero counts for tied-time-only
            # data, while others raise NoComparablePairException. Handle both.
            with np.errstate(invalid="ignore", divide="ignore"):
                result = concordance_index_censored(event, times, block[column].to_numpy())
            # sksurv's counts are concordant, discordant, tied risk, and tied time.
            # Tied-time event/censor pairs are already included in the first three counts.
            count = int(result[1] + result[2] + result[3])
            if n_pairs is None:
                n_pairs = count
                if n_pairs == 0:
                    return zeros, 0
            elif count != n_pairs:
                raise RuntimeError("Paired C-indices unexpectedly used different comparable pairs")
            numerators.append(float(result[1] + 0.5 * result[3]))
    except NoComparablePairException:
        return zeros, 0
    return numerators, n_pairs


def _block_pair_counts(block: pl.DataFrame) -> tuple[float, float, int]:
    """Return two concordance numerators on one common set of comparable pairs."""
    (text, comparator), n_pairs = _block_concordance(block, ["text_score", "comparator_score"])
    return text, comparator, n_pairs


def evaluate_comparison(
    paired: pl.DataFrame,
    *,
    scheme: str,
    event: str,
    comparator: str,
    min_patients: int = 20,
    min_events: int = 5,
) -> pl.DataFrame:
    """Evaluate already matched, validated rows; retain ineligible cancer strata."""
    if min_patients < 2 or min_events < 1:
        raise ValueError("min_patients must be >= 2 and min_events must be >= 1")
    rows = []
    for (cancer_type,), stratum in paired.group_by("cancer_type", maintain_order=True):
        n_events = int(stratum["event_flag"].sum())
        numerator_text = numerator_comparator = 0.0
        n_pairs = n_blocks = None
        if stratum.height < min_patients:
            status = "too_few_patients"
        elif n_events < min_events:
            status = "too_few_events"
        else:
            n_pairs = n_blocks = 0
            for _, block in stratum.group_by("text_fold", "comparator_fold"):
                text_count, comparator_count, count = _block_pair_counts(block)
                numerator_text += text_count
                numerator_comparator += comparator_count
                n_pairs += count
                n_blocks += int(count > 0)
            status = "ok" if n_pairs else "no_comparable_pairs"
        text_cindex = numerator_text / n_pairs if status == "ok" else None
        comparator_cindex = numerator_comparator / n_pairs if status == "ok" else None
        rows.append({
            "scheme": scheme, "event": event, "cancer_type": cancer_type,
            "comparator": comparator, "text_cindex": text_cindex,
            "comparator_cindex": comparator_cindex,
            "delta_cindex": text_cindex - comparator_cindex if status == "ok" else None,
            "n_patients": stratum.height, "n_events": n_events,
            "n_comparable_pairs": n_pairs, "n_fold_blocks": n_blocks,
            "min_patients_required": min_patients, "min_events_required": min_events,
            "status": status,
        })
    return pl.DataFrame(rows, schema=RESULT_SCHEMA).sort("cancer_type")


def _matched_modalities(
    scores: dict[str, pl.DataFrame], outcomes: pl.DataFrame, cancer: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Patients with every modality's valid score and fold, a valid outcome and a
    cancer label; also returns the number of scored patients excluded."""
    frames = list(scores.values())
    joined = reduce(lambda left, right: left.join(right, on="DFCI_MRN", how="inner", validate="1:1"),
                    frames)
    n_scored = max(frame.height for frame in frames)
    joined = joined.join(outcomes, on="DFCI_MRN", how="inner", validate="1:1")
    joined = joined.join(cancer, on="DFCI_MRN", how="left", validate="1:1")
    valid = (pl.col("time").is_finite() & (pl.col("time") > 0)
             & pl.col("event_flag").is_in([0.0, 1.0])
             & pl.col("cancer_type").is_not_null() & (pl.col("cancer_type") != ""))
    for modality in scores:
        valid = valid & pl.col(f"{modality}_score").is_finite() & _valid_fold(f"{modality}_fold")
    joined = joined.filter(valid.fill_null(False)).with_columns(
        pl.col(f"{modality}_fold").cast(pl.Int64) for modality in scores
    )
    return joined, n_scored - joined.height


def evaluate_modalities(
    joint: pl.DataFrame,
    *,
    scheme: str,
    event: str,
    modalities: list[str],
    min_patients: int = 20,
    min_events: int = 5,
) -> pl.DataFrame:
    """Per-cancer C-index for every modality on the same patients and pairs.

    Blocks are joint over all modalities' outer folds, so every compared pair is
    scored by one fitted model per modality, and all modalities share the pairs.
    """
    if min_patients < 2 or min_events < 1:
        raise ValueError("min_patients must be >= 2 and min_events must be >= 1")
    score_cols = [f"{m}_score" for m in modalities]
    fold_cols = [f"{m}_fold" for m in modalities]
    rows = []
    for (cancer_type,), stratum in joint.group_by("cancer_type", maintain_order=True):
        n_events = int(stratum["event_flag"].sum())
        numerators = [0.0] * len(modalities)
        n_pairs = n_blocks = None
        if stratum.height < min_patients:
            status = "too_few_patients"
        elif n_events < min_events:
            status = "too_few_events"
        else:
            n_pairs = n_blocks = 0
            for _, block in stratum.group_by(fold_cols):
                counts, count = _block_concordance(block, score_cols)
                numerators = [a + b for a, b in zip(numerators, counts)]
                n_pairs += count
                n_blocks += int(count > 0)
            status = "ok" if n_pairs else "no_comparable_pairs"
        for modality, numerator in zip(modalities, numerators):
            rows.append({
                "scheme": scheme, "event": event, "cancer_type": cancer_type,
                "modality": modality,
                "cindex": numerator / n_pairs if status == "ok" else None,
                "n_patients": stratum.height, "n_events": n_events,
                "n_comparable_pairs": n_pairs, "n_fold_blocks": n_blocks,
                "min_patients_required": min_patients, "min_events_required": min_events,
                "status": status,
            })
    return pl.DataFrame(rows, schema=MODALITY_RESULT_SCHEMA).sort("cancer_type", "modality")


def _evaluate_endpoint(task: tuple, context: dict) -> dict[str, list]:
    """Evaluate one (comparison, scheme, event) endpoint task.

    Pure function of its inputs so it can run in a worker process: returns the
    endpoint's audit rows, comparison metrics, source counts and shared-cohort
    modality metrics instead of appending to shared state.
    """
    comparison, scheme, event, directory, comparators = task
    cancer = context["cancer"]
    min_patients, min_events = context["min_patients"], context["min_events"]
    out: dict[str, list] = {"audit": [], "results": [], "counts": [], "modalities": []}
    audit = out["audit"]
    source_path, columns, full, modality_cohort = context["sources"][scheme]
    missing = {"DFCI_MRN", event, f"tt_{event}"} - columns
    if missing:
        _audit(audit, comparison, scheme, event, "", "missing_outcome_columns",
               ", ".join(sorted(missing)))
        return out
    try:
        outcomes = _read_outcomes(pl.scan_parquet(source_path), event)
    except _READ_ERRORS as exc:
        _audit(audit, comparison, scheme, event, "", "invalid_text_or_outcome_input", str(exc))
        return out
    cohort = full if comparison == "fig2" else modality_cohort
    if cohort is not None:
        event_counts = _pre_evaluation_counts(
            outcomes, cohort, scheme=scheme, event=event,
            min_patients=min_patients, min_events=min_events,
            cancer_types=full.select("cancer_type"),
        )
        out["counts"].append(event_counts)
        if not event_counts["eligible"].any():
            _audit(audit, comparison, scheme, event, "", "no_eligible_cancer_types",
                   "Source-cohort patient/event counts below thresholds; skipped prediction reads")
            return out
    # If membership sources are unavailable, score-derived
    # comparisons remain usable; their source counts are unknown.
    # Otherwise restrict evaluation to the source cohort counted
    # above, so its counts are genuine upper bounds.
    evaluation_cohort = full if cohort is None else cohort
    outcomes = outcomes.join(
        _event_cohort(evaluation_cohort, event).select("DFCI_MRN"),
        on="DFCI_MRN", how="inner", validate="1:1",
    )
    try:
        text_scores = _load_scores(directory / "text_risk_scores.csv", "text", "text")
    except _READ_ERRORS as exc:
        _audit(audit, comparison, scheme, event, "", "invalid_text_or_outcome_input", str(exc))
        return out
    modality_scores = {"text": text_scores}
    for comparator in comparators:
        try:
            other = _load_scores(directory / f"{comparator}_risk_scores.csv", comparator, "comparator")
        except _READ_ERRORS as exc:
            _audit(audit, comparison, scheme, event, comparator, "invalid_comparator_input", str(exc))
            continue
        modality_scores[comparator] = other.rename({
            "comparator_score": f"{comparator}_score",
            "comparator_fold": f"{comparator}_fold",
        })
        paired, exclusions = _matched_patients(text_scores, other, outcomes, cancer)
        for reason, count in exclusions.items():
            if count:
                _audit(audit, comparison, scheme, event, comparator, reason,
                       "Excluded before matched within-cancer evaluation", count)
        if paired.is_empty():
            _audit(audit, comparison, scheme, event, comparator, "no_matched_patients",
                   "No patients have valid paired predictions, outcomes, and cancer labels")
            continue
        metrics = evaluate_comparison(
            paired, scheme=scheme, event=event, comparator=comparator,
            min_patients=min_patients, min_events=min_events,
        )
        out["results"].append(metrics)
        n_ok = metrics.filter(pl.col("status") == "ok").height
        _audit(audit, comparison, scheme, event, comparator, "evaluated",
               f"{n_ok}/{metrics.height} cancer strata eligible; "
               f"min_patients={min_patients}; min_events={min_events}", paired.height)
        for row in metrics.filter(pl.col("status") != "ok").iter_rows(named=True):
            _audit(audit, comparison, scheme, event, comparator, row["status"],
                   f"cancer_type={row['cancer_type']}; n_events={row['n_events']}; "
                   f"n_comparable_pairs={row['n_comparable_pairs']}", row["n_patients"])
    if comparison == "fig3":
        _evaluate_all_modalities(
            modality_scores, outcomes, cancer, audit, out["modalities"],
            scheme=scheme, event=event,
            min_patients=min_patients, min_events=min_events,
        )
    return out


# Set once per worker process by _init_worker so tasks need not re-send the
# cancer labels and cohort tables.
_WORKER_CONTEXT: dict | None = None


def _init_worker(context: dict) -> None:
    global _WORKER_CONTEXT
    warnings.simplefilter("ignore")
    _WORKER_CONTEXT = context


def _evaluate_endpoint_in_worker(task: tuple) -> dict[str, list]:
    return _evaluate_endpoint(task, _WORKER_CONTEXT)


def _run_endpoint_tasks(tasks: list[tuple], context: dict, pool, progress) -> list[dict[str, list]]:
    """Evaluate tasks serially or on `pool`; results are in task order either way."""
    if pool is None:
        outputs = []
        for task in tasks:
            try:
                outputs.append(_evaluate_endpoint(task, context))
            finally:
                # Missing files, sparse endpoints and all modality comparisons
                # together still count as one completed endpoint task.
                progress.update(1)
        return outputs
    futures = [pool.submit(_evaluate_endpoint_in_worker, task) for task in tasks]
    for future in as_completed(futures):
        progress.update(1)
    return [future.result() for future in futures]


def _prepare_within_cancer(
    *, min_patients: int, min_events: int, show_progress: bool, n_jobs: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    audit: list[dict] = []
    results: dict[str, list[pl.DataFrame]] = {"fig2": [], "fig3": []}
    modality_results: list[pl.DataFrame] = []
    counts: dict[str, list[pl.DataFrame]] = {"fig2": [], "fig3": []}
    try:
        cancer = _load_cancer_types(Path(FEATURE_PATH) / "cancer_type_df.csv.gz")
    except _READ_ERRORS as exc:
        _audit(audit, "all", "", "", "", "invalid_cancer_input", str(exc))
        cancer = None

    sources = {}
    common_ids = None
    if cancer is not None:
        try:
            common_ids = _load_modality_cohort_ids()
        except _READ_ERRORS as exc:
            _audit(audit, "fig3", "", "", "", "unavailable_modality_cohort_counts", str(exc))
        for scheme in SCHEMES:
            try:
                source_path = Path(SURV_PATH) / embedding_file(scheme)
                source = pl.scan_parquet(source_path)
                columns = set(source.collect_schema().names())
                ids = _validate_ids(source.select("DFCI_MRN").collect(), f"{scheme} cohort")
                full = ids.join(cancer, on="DFCI_MRN", how="inner", validate="1:1").filter(
                    pl.col("cancer_type").is_not_null() & (pl.col("cancer_type") != "")
                )
                modality = (full.join(common_ids, on="DFCI_MRN", how="inner", validate="1:1")
                            if common_ids is not None else None)
                sources[scheme] = (source_path, columns, full, modality)
            except _READ_ERRORS as exc:
                _audit(audit, "all", scheme, "", "", "missing_outcome_input", str(exc))

    context = {"cancer": cancer, "sources": sources,
               "min_patients": min_patients, "min_events": min_events}
    all_tasks = {}
    for comparison, subdir, comparators, description in (
        ("fig2", "full_cohort_risk_scores", ["base"], "Full cohort"),
        ("fig3", "held_out_risk_scores", [m for m in MODALITY_ORDER if m != "text"], "Modality cohort"),
    ):
        tasks = []
        for scheme, (_, columns, _, _) in sources.items():
            root = Path(scheme_results_dir(scheme)) / subdir
            # Count every source endpoint, including unfinished model runs, and
            # audit orphan risk directories rather than silently discarding them.
            directories = {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
            events = {c[3:] for c in columns if c.startswith("tt_")}
            tasks.extend((comparison, scheme, event, root / event, comparators)
                         for event in sorted(events | directories))
        all_tasks[comparison] = (tasks, description)

    n_workers = min(resolve_workers(n_jobs), max((len(t) for t, _ in all_tasks.values()), default=1))
    with process_pool(n_workers, initializer=_init_worker, initargs=(context,)) as pool:
        for comparison, (tasks, description) in all_tasks.items():
            with tqdm(total=len(tasks), desc=description, unit="event", leave=True,
                      dynamic_ncols=True, disable=not show_progress) as progress:
                for output in _run_endpoint_tasks(tasks, context, pool, progress):
                    audit.extend(output["audit"])
                    results[comparison].extend(output["results"])
                    counts[comparison].extend(output["counts"])
                    modality_results.extend(output["modalities"])
    frames = [
        pl.concat(results[key]).sort("scheme", "event", "cancer_type", "comparator")
        if results[key] else pl.DataFrame(schema=RESULT_SCHEMA)
        for key in ("fig2", "fig3")
    ]
    count_frames = [
        pl.concat(counts[key]).sort("scheme", "event", "cancer_type")
        if counts[key] else pl.DataFrame(schema=COUNT_SCHEMA)
        for key in ("fig2", "fig3")
    ]
    modality_frame = (
        pl.concat(modality_results).sort("scheme", "event", "cancer_type", "modality")
        if modality_results else pl.DataFrame(schema=MODALITY_RESULT_SCHEMA)
    )
    return (frames[0], frames[1], pl.DataFrame(audit, schema=AUDIT_SCHEMA), *count_frames,
            modality_frame)


def _evaluate_all_modalities(
    modality_scores: dict[str, pl.DataFrame], outcomes: pl.DataFrame, cancer: pl.DataFrame,
    audit: list[dict], results: list[pl.DataFrame], *, scheme: str, event: str,
    min_patients: int, min_events: int,
) -> None:
    """Append the shared-cohort modality C-indices for one endpoint, or audit why not."""
    modalities = [m for m in MODALITY_ORDER if m in modality_scores]
    missing = [m for m in MODALITY_ORDER if m not in modality_scores]
    if missing:
        _audit(audit, "fig3_modalities", scheme, event, "", "incomplete_modalities",
               "Missing scores: " + ", ".join(missing))
        return
    joint, n_excluded = _matched_modalities(
        {m: modality_scores[m] for m in modalities}, outcomes, cancer
    )
    if n_excluded:
        _audit(audit, "fig3_modalities", scheme, event, "", "excluded_before_joint_evaluation",
               "Scored patients lacking any modality score, fold, outcome or cancer label", n_excluded)
    if joint.is_empty():
        _audit(audit, "fig3_modalities", scheme, event, "", "no_matched_patients",
               "No patients have every modality's prediction, an outcome and a cancer label")
        return
    metrics = evaluate_modalities(
        joint, scheme=scheme, event=event, modalities=modalities,
        min_patients=min_patients, min_events=min_events,
    )
    results.append(metrics)
    n_ok = metrics.filter(pl.col("status") == "ok")["cancer_type"].n_unique()
    _audit(audit, "fig3_modalities", scheme, event, "", "evaluated",
           f"{n_ok}/{metrics['cancer_type'].n_unique()} cancer strata eligible", joint.height)


def prepare_within_cancer(
    *, min_patients: int = 20, min_events: int = 5, show_progress: bool = True,
    n_jobs: int | None = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return Fig2 metrics, Fig3 metrics, audit, full counts, modality counts and
    shared-cohort per-modality C-indices.

    Warning suppression is scoped to this run, including all file reads. Audit
    entries are persisted instead of printed. Exceptions still propagate.
    Endpoints are evaluated on n_jobs worker processes (None: FIGURE_PREP_N_JOBS,
    then the SLURM/CPU allocation); outputs are identical to the serial run.
    """
    if min_patients < 2 or min_events < 1:
        raise ValueError("min_patients must be >= 2 and min_events must be >= 1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _prepare_within_cancer(
            min_patients=min_patients, min_events=min_events, show_progress=show_progress,
            n_jobs=n_jobs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-patients", type=int, default=20)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="worker processes (default: FIGURE_PREP_N_JOBS, then the CPU allocation)")
    args = parser.parse_args()
    if args.min_patients < 2 or args.min_events < 1:
        parser.error("--min-patients must be >= 2 and --min-events must be >= 1")
    frames = prepare_within_cancer(min_patients=args.min_patients, min_events=args.min_events,
                                   n_jobs=args.n_jobs)
    names = (
        "fig2_within_cancer_cindex.csv", "fig3_within_cancer_cindex.csv",
        "within_cancer_audit.csv", "fig2_within_cancer_event_counts.csv",
        "fig3_within_cancer_event_counts.csv", "fig3_within_cancer_modality_cindex.csv",
    )
    # CSV-only output avoids importing figures.io's matplotlib runtime or its
    # warning/print messages. The only normal terminal output is the two bars.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = Path(FIGURE_DATA_DIR)
        output.mkdir(parents=True, exist_ok=True)
        for frame, name in zip(frames, names):
            frame.write_csv(output / name)


if __name__ == "__main__":
    main()
