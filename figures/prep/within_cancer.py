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

Cancer rows below the sample/event thresholds or lacking comparable pairs remain
in the performance CSV with null metrics and an explanatory status. Missing or
malformed inputs and row exclusions are recorded in the audit CSV and the log.
Only status == 'ok' rows should be plotted. n_fold_blocks counts blocks with
at least one comparable pair; n_patients and n_events describe the full matched
valid cohort, including patients in blocks with no comparable pairs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored

from config import FEATURE_PATH, SURV_PATH
from figures.io import save_figure_data
from schemes import embedding_file, scheme_results_dir
from shared.palette import MODALITY_ORDER

logger = logging.getLogger(__name__)

SCHEMES = ("death_met", "icd3_post", "icd4_post", "phecode_post")
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
    log = logger.info if status == "evaluated" else logger.warning
    log("[%s/%s/%s/%s] %s: %s (%s rows)",
        comparison, scheme, event, comparator, status, detail, n_rows)


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
        path, columns=["DFCI_MRN", "CANCER_TYPE"],
        schema_overrides={"DFCI_MRN": pl.String, "CANCER_TYPE": pl.String},
    )
    return _validate_ids(cancer, str(path)).with_columns(
        pl.col("CANCER_TYPE").str.strip_chars().alias("cancer_type")
    ).select("DFCI_MRN", "cancer_type")


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


def _block_pair_counts(block: pl.DataFrame) -> tuple[float, float, int]:
    """Return two concordance numerators on one common set of comparable pairs."""
    if block.height < 2 or block["event_flag"].sum() == 0:
        return 0.0, 0.0, 0
    event = block["event_flag"].cast(pl.Boolean).to_numpy()
    times = block["time"].to_numpy()
    try:
        # Some sksurv releases return NaN plus zero counts for tied-time-only
        # data, while others raise NoComparablePairException. Handle both.
        with np.errstate(invalid="ignore", divide="ignore"):
            text = concordance_index_censored(event, times, block["text_score"].to_numpy())
        n_pairs = int(text[1] + text[2] + text[3])
        if n_pairs == 0:
            return 0.0, 0.0, 0
        comparator = concordance_index_censored(
            event, times, block["comparator_score"].to_numpy()
        )
    except NoComparablePairException:
        return 0.0, 0.0, 0
    # sksurv's counts are concordant, discordant, tied risk, and tied time.
    # Tied-time event/censor pairs are already included in the first three counts.
    if n_pairs != int(comparator[1] + comparator[2] + comparator[3]):
        raise RuntimeError("Paired C-indices unexpectedly used different comparable pairs")
    return float(text[1] + 0.5 * text[3]), float(comparator[1] + 0.5 * comparator[3]), n_pairs


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
        n_pairs = n_blocks = 0
        for _, block in stratum.group_by("text_fold", "comparator_fold"):
            text_count, comparator_count, count = _block_pair_counts(block)
            numerator_text += text_count
            numerator_comparator += comparator_count
            n_pairs += count
            n_blocks += int(count > 0)
        if stratum.height < min_patients:
            status = "too_few_patients"
        elif n_events < min_events:
            status = "too_few_events"
        elif n_pairs == 0:
            status = "no_comparable_pairs"
        else:
            status = "ok"
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


def prepare_within_cancer(
    *, min_patients: int = 20, min_events: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return text/base data, text/modality data, and a missing-input/exclusion audit."""
    if min_patients < 2 or min_events < 1:
        raise ValueError("min_patients must be >= 2 and min_events must be >= 1")
    audit: list[dict] = []
    results: dict[str, list[pl.DataFrame]] = {"fig2": [], "fig3": []}
    try:
        cancer = _load_cancer_types(Path(FEATURE_PATH) / "cancer_type_df.csv.gz")
    except _READ_ERRORS as exc:
        _audit(audit, "all", "", "", "", "invalid_cancer_input", str(exc))
        return (pl.DataFrame(schema=RESULT_SCHEMA), pl.DataFrame(schema=RESULT_SCHEMA),
                pl.DataFrame(audit, schema=AUDIT_SCHEMA))

    for scheme in SCHEMES:
        outcome_path = Path(SURV_PATH) / embedding_file(scheme)
        # Source construction is lazy; schema discovery is the first file access.
        try:
            source = pl.scan_parquet(outcome_path)
            columns = set(source.collect_schema().names())
        except _READ_ERRORS as exc:
            _audit(audit, "all", scheme, "", "", "missing_outcome_input", str(exc))
            continue
        for comparison, subdir, comparators in (
            ("fig2", "full_cohort_risk_scores", ["base"]),
            ("fig3", "held_out_risk_scores", [m for m in MODALITY_ORDER if m != "text"]),
        ):
            root = Path(scheme_results_dir(scheme)) / subdir
            directories = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
            if not directories:
                _audit(audit, comparison, scheme, "", "", "missing_risk_input", str(root))
                continue
            for directory in directories:
                event = directory.name
                missing = {"DFCI_MRN", event, f"tt_{event}"} - columns
                if missing:
                    _audit(audit, comparison, scheme, event, "", "missing_outcome_columns",
                           ", ".join(sorted(missing)))
                    continue
                try:
                    outcomes = _read_outcomes(source, event)
                    text = _load_scores(directory / "text_risk_scores.csv", "text", "text")
                except _READ_ERRORS as exc:
                    _audit(audit, comparison, scheme, event, "", "invalid_text_or_outcome_input", str(exc))
                    continue
                for modality in comparators:
                    try:
                        other = _load_scores(directory / f"{modality}_risk_scores.csv", modality, "comparator")
                    except _READ_ERRORS as exc:
                        _audit(audit, comparison, scheme, event, modality, "invalid_comparator_input", str(exc))
                        continue
                    paired, exclusions = _matched_patients(text, other, outcomes, cancer)
                    for reason, count in exclusions.items():
                        if count:
                            _audit(audit, comparison, scheme, event, modality, reason,
                                   "Excluded before matched within-cancer evaluation", count)
                    if paired.is_empty():
                        _audit(audit, comparison, scheme, event, modality, "no_matched_patients",
                               "No patients have valid paired predictions, outcomes, and cancer labels")
                        continue
                    metrics = evaluate_comparison(
                        paired, scheme=scheme, event=event, comparator=modality,
                        min_patients=min_patients, min_events=min_events,
                    )
                    results[comparison].append(metrics)
                    n_ok = metrics.filter(pl.col("status") == "ok").height
                    _audit(audit, comparison, scheme, event, modality, "evaluated",
                           f"{n_ok}/{metrics.height} cancer strata eligible; "
                           f"min_patients={min_patients}; min_events={min_events}", paired.height)
                    for row in metrics.filter(pl.col("status") != "ok").iter_rows(named=True):
                        _audit(audit, comparison, scheme, event, modality, row["status"],
                               f"cancer_type={row['cancer_type']}; n_events={row['n_events']}; "
                               f"n_comparable_pairs={row['n_comparable_pairs']}", row["n_patients"])
    frames = [
        pl.concat(results[key]).sort("scheme", "event", "cancer_type", "comparator")
        if results[key] else pl.DataFrame(schema=RESULT_SCHEMA)
        for key in ("fig2", "fig3")
    ]
    return frames[0], frames[1], pl.DataFrame(audit, schema=AUDIT_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-patients", type=int, default=20)
    parser.add_argument("--min-events", type=int, default=5)
    args = parser.parse_args()
    if args.min_patients < 2 or args.min_events < 1:
        parser.error("--min-patients must be >= 2 and --min-events must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fig2, fig3, audit = prepare_within_cancer(
        min_patients=args.min_patients, min_events=args.min_events
    )
    save_figure_data(fig2, "fig2_within_cancer_cindex.csv")
    save_figure_data(fig3, "fig3_within_cancer_cindex.csv")
    save_figure_data(audit, "within_cancer_audit.csv")


if __name__ == "__main__":
    main()
