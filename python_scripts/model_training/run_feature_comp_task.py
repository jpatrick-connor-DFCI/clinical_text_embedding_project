"""Run Feature Comp Task script for model training workflows."""

import argparse
import os

from embed_surv_utils import run_grid_CoxPH_parallel

from slurm_array_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    _get_n_jobs,
    filter_event_rows,
    get_events_from_df,
    get_output_dir,
    load_feature_modalities_df,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one feature-comparison modality for one endpoint event.")
    parser.add_argument("--scheme", required=True, choices=["icd3", "icd4", "phecode", "death_met"])
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--modality",
        required=True,
        choices=["stage", "treatment", "labs", "somatic", "prs", "text"],
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    full_prediction_df, type_cols, _, modality_cfg, _ = load_feature_modalities_df(args.scheme)
    events = get_events_from_df(full_prediction_df)
    if args.event not in events:
        raise ValueError(
            f"Event '{args.event}' not found for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )
    if args.modality not in modality_cfg:
        raise ValueError(f"Unsupported modality '{args.modality}'.")

    out_dir = os.path.join(get_output_dir(args.scheme, "feature_comps"), args.event)
    os.makedirs(out_dir, exist_ok=True)
    test_fp = os.path.join(out_dir, f"{args.modality}_test.csv")
    val_fp = os.path.join(out_dir, f"{args.modality}_val.csv")
    if (not args.overwrite) and os.path.exists(test_fp) and os.path.exists(val_fp):
        print(f"[skip] Existing outputs found for {args.scheme}:{args.event}:{args.modality}")
        return

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.empty:
        print(f"[skip] No rows with tt_{args.event} > 0 for {args.scheme}:{args.event}")
        return

    base_vars = ["GENDER", "AGE_AT_TREATMENTSTART"]
    cfg = modality_cfg[args.modality]
    n_jobs = _get_n_jobs(args.n_jobs)

    test_df, val_df, _ = run_grid_CoxPH_parallel(
        event_pred_df,
        base_vars + type_cols,
        cfg["continuous_vars"],
        cfg["penalized_cols"],
        DEFAULT_L1_RATIOS,
        DEFAULT_ALPHAS,
        pca_config=cfg["pca_config"],
        event_col=args.event,
        tstop_col=f"tt_{args.event}",
        max_iter=args.max_iter,
        n_jobs=n_jobs,
        backend=args.backend,
    )
    test_df.to_csv(test_fp, index=False)
    val_df.to_csv(val_fp, index=False)
    print(f"[done] {args.scheme}:{args.event}:{args.modality} -> {out_dir}")


if __name__ == "__main__":
    main()
