"""Generate 5-fold held-out risk scores for the full-cohort text and base models.

Uses nested CV for the penalized text model so each patient's outer-fold score
is produced with hyperparameters selected without that patient's outcome. The
base model is unpenalized. Outputs per-patient risk scores and outer-fold IDs.
"""

import argparse
import json
import os
import time

import polars as pl

from anchors import ANCHORS, DEFAULT_ANCHOR, age_col
from config import SURV_PATH
from schemes import SCHEMES, get_output_dir
from survival import get_heldout_risk_scores_CoxPH, get_nested_heldout_risk_scores_CoxPH

from pipelines.training.slurm_array_utils import (
    _get_n_jobs,
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    adaptive_low_alphas_for,
    build_full_prediction_df,
    filter_event_rows,
    parse_float_list,
    validate_cox_inputs,
)

SKIP_REPORT_DIR = os.path.join(SURV_PATH, "results", "skipped_events")


def _write_skip_report(scheme: str, event: str, run_type: str, reason: str) -> None:
    os.makedirs(SKIP_REPORT_DIR, exist_ok=True)
    report_fp = os.path.join(SKIP_REPORT_DIR, f"{scheme}_{run_type}_skipped.jsonl")
    entry = {"scheme": scheme, "event": event, "run_type": run_type, "reason": reason}
    with open(report_fp, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-cohort held-out risk scores (text and base) for a single endpoint."
    )
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEMES.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=sorted(ANCHORS.keys()))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alphas", type=parse_float_list, default=DEFAULT_ALPHAS)
    parser.add_argument("--l1-ratios", type=parse_float_list, default=DEFAULT_L1_RATIOS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    run_type = "full_cohort_risk_scores"

    # Output dir for risk scores
    risk_out_dir = os.path.join(get_output_dir(args.scheme, run_type, args.anchor), args.event)
    os.makedirs(risk_out_dir, exist_ok=True)
    text_risk_fp = os.path.join(risk_out_dir, "text_risk_scores.csv")
    base_risk_fp = os.path.join(risk_out_dir, "base_risk_scores.csv")
    run_text = args.overwrite or not os.path.exists(text_risk_fp)
    run_base = args.overwrite or not os.path.exists(base_risk_fp)
    if not run_text and not run_base:
        print(f"[skip] Existing risk scores found for {args.scheme}:{args.event}")
        return

    # Build cohort identically to training
    full_prediction_df, type_cols, embed_cols, events = build_full_prediction_df(args.scheme, args.anchor)
    if args.event not in events:
        raise ValueError(
            f"Event '{args.event}' not found for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.is_empty():
        reason = f"No rows with tt_{args.event} > 0"
        print(f"[skip-data] {args.scheme}:{args.event} — {reason}")
        _write_skip_report(args.scheme, args.event, run_type, reason)
        raise RuntimeError(reason)

    anchor_age_col = age_col(args.anchor)
    base_vars = ["GENDER", anchor_age_col]
    all_feature_cols = base_vars + type_cols + embed_cols
    label = f"{args.scheme}:{args.event}"
    try:
        event_pred_df, dropped_cols = validate_cox_inputs(
            event_pred_df, args.event, f"tt_{args.event}", all_feature_cols, label=label,
        )
    except ValueError as e:
        reason = str(e)
        print(f"[skip-data] {label} — {reason}")
        _write_skip_report(args.scheme, args.event, run_type, reason)
        raise RuntimeError(reason) from e
    type_cols = [c for c in type_cols if c not in dropped_cols]
    embed_cols = [c for c in embed_cols if c not in dropped_cols]
    all_feature_cols = [c for c in all_feature_cols if c not in dropped_cols]
    n_before = len(event_pred_df)
    required_cols = all_feature_cols + [args.event, f"tt_{args.event}"]
    event_pred_df = event_pred_df.filter(
        pl.all_horizontal([
            pl.col(c).cast(pl.Float64, strict=False).is_finite()
            for c in required_cols
        ])
    )
    n_dropped = n_before - len(event_pred_df)
    if n_dropped > 0:
        print(f"{label} Dropped {n_dropped}/{n_before} rows with NaN values")
    n_jobs = _get_n_jobs(args.n_jobs)

    if run_text:
        t0 = time.time()
        text_scores = get_nested_heldout_risk_scores_CoxPH(
            event_pred_df,
            base_vars + type_cols,
            [anchor_age_col] + embed_cols,
            embed_cols,
            args.l1_ratios,
            args.alphas,
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
            n_splits=args.n_splits,
            n_jobs=n_jobs,
            backend=args.backend,
            adaptive_low_alphas=adaptive_low_alphas_for(args.alphas),
        ).rename({"risk_score": "text_risk_score"})
        text_scores.write_csv(text_risk_fp)
        print(f"[time] {label} text risk: {(time.time() - t0) / 60:.1f}m ({len(text_scores)} patients)")
    else:
        print(f"[skip] {label} text risk scores already exist")

    if run_base:
        t0 = time.time()
        base_scores = get_heldout_risk_scores_CoxPH(
            event_pred_df,
            base_vars + type_cols,
            [anchor_age_col],
            [],
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
            penalized=False,
            n_splits=args.n_splits,
            n_jobs=n_jobs,
            backend=args.backend,
        ).rename({"risk_score": "base_risk_score"})
        base_scores.write_csv(base_risk_fp)
        print(f"[time] {label} base risk: {(time.time() - t0) / 60:.1f}m ({len(base_scores)} patients)")
    else:
        print(f"[skip] {label} base risk scores already exist")

    print(f"[done] {args.scheme}:{args.event} -> {risk_out_dir}")


if __name__ == "__main__":
    main()
