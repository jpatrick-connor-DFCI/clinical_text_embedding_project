"""Run Full Cohort Event script for model training workflows."""

import argparse
import json
import os
import time

from anchors import ANCHORS, DEFAULT_ANCHOR, age_col
from config import SURV_PATH
from schemes import SCHEMES, get_output_dir
from survival import run_base_CoxPH, run_grid_CoxPH_parallel

from pipelines.training.slurm_array_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    _get_n_jobs,
    build_full_prediction_df,
    filter_event_rows,
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
    parser = argparse.ArgumentParser(description="Run one full-cohort model for a single endpoint event.")
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEMES.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=sorted(ANCHORS.keys()))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    full_prediction_df, type_cols, embed_cols, events = build_full_prediction_df(args.scheme, args.anchor)
    if args.event not in events:
        raise ValueError(
            f"Event '{args.event}' not found for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )

    out_dir = os.path.join(get_output_dir(args.scheme, "full_cohort", args.anchor), args.event)
    os.makedirs(out_dir, exist_ok=True)

    text_test_fp = os.path.join(out_dir, "text_test.csv")
    text_val_fp = os.path.join(out_dir, "text_val.csv")
    base_test_fp = os.path.join(out_dir, "base_test.csv")
    base_val_fp = os.path.join(out_dir, "base_val.csv")
    run_text = args.overwrite or not (os.path.exists(text_test_fp) and os.path.exists(text_val_fp))
    run_base = args.overwrite or not (os.path.exists(base_test_fp) and os.path.exists(base_val_fp))
    if not run_text and not run_base:
        print(f"[skip] Existing outputs found for {args.scheme}:{args.event}")
        return

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.empty:
        reason = f"No rows with tt_{args.event} > 0"
        print(f"[skip-data] {args.scheme}:{args.event} — {reason}")
        _write_skip_report(args.scheme, args.event, "full_cohort", reason)
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
        _write_skip_report(args.scheme, args.event, "full_cohort", reason)
        raise RuntimeError(reason) from e
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
        t0 = time.time()
        text_test, text_val, _ = run_grid_CoxPH_parallel(
            event_pred_df,
            base_vars + type_cols,
            [anchor_age_col] + embed_cols,
            embed_cols,
            DEFAULT_L1_RATIOS,
            DEFAULT_ALPHAS,
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
            n_jobs=n_jobs,
            backend=args.backend,
        )
        text_test.to_csv(text_test_fp, index=False)
        text_val.to_csv(text_val_fp, index=False)
        print(f"[time] {label} text model: {(time.time() - t0) / 60:.1f}m")
    else:
        print(f"[skip] {label} text model already exists")

    if run_base:
        t0 = time.time()
        base_results = run_base_CoxPH(
            event_pred_df,
            base_vars + type_cols,
            [anchor_age_col],
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
        )
        base_results[base_results["eval_data"] == "test_data"].drop(columns="eval_data").to_csv(base_test_fp, index=False)
        base_results[base_results["eval_data"] == "cv_data"].drop(columns="eval_data").to_csv(base_val_fp, index=False)
        print(f"[time] {label} base model: {(time.time() - t0) / 60:.1f}m")
    else:
        print(f"[skip] {label} base model already exists")

    print(f"[done] {args.scheme}:{args.event} -> {out_dir}")


if __name__ == "__main__":
    main()
