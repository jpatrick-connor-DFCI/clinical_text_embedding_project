"""Generate 5-fold held-out risk scores for the full-cohort text and base models.

Consumes the CV grid (`text_val.csv.gz`) produced by `run_full_cohort_event.py`
to pick best hyperparameters for the text model. The base model is unpenalized
(no hyperparameters needed). Outputs per-patient risk scores so the two models
can be compared on the same cohort (e.g. stratified KM curves).
"""

import argparse
import json
import os
import time

from embed_surv_utils import get_heldout_risk_scores_CoxPH

from slurm_array_utils import (
    SCHEME_CONFIG,
    SURV_PATH,
    _get_n_jobs,
    build_full_prediction_df,
    filter_event_rows,
    get_best_hparams,
    get_output_dir,
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
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEME_CONFIG.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    run_type = "full_cohort_risk_scores"

    # Output dir for risk scores
    risk_out_dir = os.path.join(get_output_dir(args.scheme, run_type), args.event)
    os.makedirs(risk_out_dir, exist_ok=True)
    text_risk_fp = os.path.join(risk_out_dir, "text_risk_scores.csv.gz")
    base_risk_fp = os.path.join(risk_out_dir, "base_risk_scores.csv.gz")
    run_text = args.overwrite or not os.path.exists(text_risk_fp)
    run_base = args.overwrite or not os.path.exists(base_risk_fp)
    if not run_text and not run_base:
        print(f"[skip] Existing risk scores found for {args.scheme}:{args.event}")
        return

    # Training outputs we depend on
    train_dir = os.path.join(get_output_dir(args.scheme, "full_cohort"), args.event)
    text_val_fp = os.path.join(train_dir, "text_val.csv.gz")
    if run_text and not os.path.exists(text_val_fp):
        reason = f"Missing training CV output {text_val_fp}; run run_full_cohort_event.py first"
        print(f"[skip-data] {args.scheme}:{args.event} — {reason}")
        _write_skip_report(args.scheme, args.event, run_type, reason)
        return

    # Build cohort identically to training
    full_prediction_df, type_cols, embed_cols, events = build_full_prediction_df(args.scheme)
    if args.event not in events:
        raise ValueError(
            f"Event '{args.event}' not found for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.empty:
        reason = f"No rows with tt_{args.event} > 0"
        print(f"[skip-data] {args.scheme}:{args.event} — {reason}")
        _write_skip_report(args.scheme, args.event, run_type, reason)
        return

    base_vars = ["GENDER", "AGE_AT_TREATMENTSTART"]
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
        return
    type_cols = [c for c in type_cols if c not in dropped_cols]
    embed_cols = [c for c in embed_cols if c not in dropped_cols]
    all_feature_cols = [c for c in all_feature_cols if c not in dropped_cols]
    n_before = len(event_pred_df)
    event_pred_df = event_pred_df.dropna(subset=all_feature_cols + [args.event, f"tt_{args.event}"])
    n_dropped = n_before - len(event_pred_df)
    if n_dropped > 0:
        print(f"{label} Dropped {n_dropped}/{n_before} rows with NaN values")
    n_jobs = _get_n_jobs(args.n_jobs)

    if run_text:
        text_l1, text_alpha = get_best_hparams(text_val_fp)
        t0 = time.time()
        text_scores = get_heldout_risk_scores_CoxPH(
            event_pred_df,
            base_vars + type_cols,
            ["AGE_AT_TREATMENTSTART"] + embed_cols,
            embed_cols,
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
            penalized=True,
            l1_ratio=text_l1,
            alpha=text_alpha,
            n_jobs=n_jobs,
            backend=args.backend,
        ).rename(columns={"risk_score": "text_risk_score"})
        text_scores.to_csv(text_risk_fp, index=False)
        print(f"[time] {label} text risk: {(time.time() - t0) / 60:.1f}m ({len(text_scores)} patients)")
    else:
        print(f"[skip] {label} text risk scores already exist")

    if run_base:
        t0 = time.time()
        base_scores = get_heldout_risk_scores_CoxPH(
            event_pred_df,
            base_vars + type_cols,
            ["AGE_AT_TREATMENTSTART"],
            [],
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
            penalized=False,
            n_jobs=n_jobs,
            backend=args.backend,
        ).rename(columns={"risk_score": "base_risk_score"})
        base_scores.to_csv(base_risk_fp, index=False)
        print(f"[time] {label} base risk: {(time.time() - t0) / 60:.1f}m ({len(base_scores)} patients)")
    else:
        print(f"[skip] {label} base risk scores already exist")

    print(f"[done] {args.scheme}:{args.event} -> {risk_out_dir}")


if __name__ == "__main__":
    main()
