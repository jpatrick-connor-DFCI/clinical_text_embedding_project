"""Run Full Cohort Event script for model training workflows."""

import argparse
import os

from embed_surv_utils import run_grid_CoxPH_parallel

from slurm_array_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    SCHEME_CONFIG,
    _get_n_jobs,
    build_full_prediction_df,
    filter_event_rows,
    get_output_dir,
    validate_cox_inputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one full-cohort model for a single endpoint event.")
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEME_CONFIG.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    full_prediction_df, type_cols, embed_cols, events = build_full_prediction_df(args.scheme)
    if args.event not in events:
        raise ValueError(
            f"Event '{args.event}' not found for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )

    out_dir = os.path.join(get_output_dir(args.scheme, "full_cohort"), args.event)
    os.makedirs(out_dir, exist_ok=True)

    text_test_fp = os.path.join(out_dir, "text_test.csv")
    text_val_fp = os.path.join(out_dir, "text_val.csv")
    base_test_fp = os.path.join(out_dir, "base_test.csv")
    base_val_fp = os.path.join(out_dir, "base_val.csv")
    all_fps = [text_test_fp, text_val_fp, base_test_fp, base_val_fp]
    if (not args.overwrite) and all(os.path.exists(fp) for fp in all_fps):
        print(f"[skip] Existing outputs found for {args.scheme}:{args.event}")
        return

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.empty:
        print(f"[skip] No rows with tt_{args.event} > 0 for {args.scheme}:{args.event}")
        return

    base_vars = ["GENDER", "AGE_AT_TREATMENTSTART"]
    all_feature_cols = base_vars + type_cols + embed_cols
    label = f"{args.scheme}:{args.event}"
    event_pred_df = validate_cox_inputs(
        event_pred_df, args.event, f"tt_{args.event}", all_feature_cols, label=label,
    )
    n_jobs = _get_n_jobs(args.n_jobs)

    text_test, text_val, _ = run_grid_CoxPH_parallel(
        event_pred_df,
        base_vars + type_cols,
        ["AGE_AT_TREATMENTSTART"] + embed_cols,
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

    base_test, base_val, _ = run_grid_CoxPH_parallel(
        event_pred_df,
        base_vars + type_cols,
        ["AGE_AT_TREATMENTSTART"],
        [],
        [1.0],
        [1e-10],
        event_col=args.event,
        tstop_col=f"tt_{args.event}",
        max_iter=args.max_iter,
        n_jobs=n_jobs,
        backend=args.backend,
    )
    base_test.to_csv(base_test_fp, index=False)
    base_val.to_csv(base_val_fp, index=False)

    print(f"[done] {args.scheme}:{args.event} -> {out_dir}")


if __name__ == "__main__":
    main()
