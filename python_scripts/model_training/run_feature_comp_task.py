"""Run Feature Comp Task script for model training workflows."""

import argparse
import os
import time

import pandas as pd

from embed_surv_utils import get_heldout_risk_scores_CoxPH, run_grid_CoxPH_parallel

from slurm_array_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    SCHEME_CONFIG,
    _get_n_jobs,
    filter_event_rows,
    get_events_from_df,
    get_output_dir,
    load_feature_modalities_df,
    validate_cox_inputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one feature-comparison modality for one endpoint event.")
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEME_CONFIG.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--modality",
        required=True,
        choices=["stage", "treatment", "labs", "somatic", "prs", "text"],
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    full_prediction_df, type_cols, _, modality_cfg, _ = load_feature_modalities_df(args.scheme, modality=args.modality)
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
    risk_dir = os.path.join(get_output_dir(args.scheme, "feature_comps"), "..", "held_out_risk_scores", args.event)
    risk_dir = os.path.normpath(risk_dir)
    risk_fp = os.path.join(risk_dir, f"{args.modality}_risk_scores.csv")
    grid_done = os.path.exists(test_fp) and os.path.exists(val_fp)
    risk_done = os.path.exists(risk_fp)
    if (not args.overwrite) and grid_done and risk_done:
        print(f"[skip] Existing outputs found for {args.scheme}:{args.event}:{args.modality}")
        return

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.empty:
        print(f"[skip] No rows with tt_{args.event} > 0 for {args.scheme}:{args.event}")
        return

    base_vars = ["GENDER", "AGE_AT_TREATMENTSTART"]
    cfg = modality_cfg[args.modality]
    all_feature_cols = base_vars + type_cols + cfg["continuous_vars"] + cfg["penalized_cols"]
    # deduplicate while preserving order
    seen = set()
    all_feature_cols = [c for c in all_feature_cols if not (c in seen or seen.add(c))]
    label = f"{args.scheme}:{args.event}:{args.modality}"
    event_pred_df, dropped_cols = validate_cox_inputs(
        event_pred_df, args.event, f"tt_{args.event}", all_feature_cols, label=label,
    )
    type_cols = [c for c in type_cols if c not in dropped_cols]
    cfg["continuous_vars"] = [c for c in cfg["continuous_vars"] if c not in dropped_cols]
    cfg["penalized_cols"] = [c for c in cfg["penalized_cols"] if c not in dropped_cols]
    all_feature_cols = [c for c in all_feature_cols if c not in dropped_cols]
    n_before = len(event_pred_df)
    # Columns with a paired _missing indicator are raw measurements deferred to per-fold
    # mean imputation (e.g. lab values). All other feature NaNs are unexpected and drop the row.
    imputable_cols = {c for c in all_feature_cols if f"{c}_missing" in event_pred_df.columns}
    drop_subset = [c for c in all_feature_cols if c not in imputable_cols] + [args.event, f"tt_{args.event}"]
    event_pred_df = event_pred_df.dropna(subset=drop_subset)
    n_dropped = n_before - len(event_pred_df)
    if n_dropped > 0:
        print(f"{label} Dropped {n_dropped}/{n_before} rows with NaN values")
    # For small feature sets the joblib overhead dominates; run serially.
    n_jobs = 1 if len(cfg["penalized_cols"]) < 50 else _get_n_jobs(args.n_jobs)

    # --- Grid search (skip if already done, reuse val results for risk scores) ---
    if (not args.overwrite) and grid_done:
        print(f"[skip] {label} grid search already done, loading val results")
        val_df = pd.read_csv(val_fp)
    else:
        t0 = time.time()
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
        print(f"[time] {label} grid search: {(time.time() - t0) / 60:.1f}m")

    # --- Generate held-out risk scores using best hyperparams ---
    os.makedirs(risk_dir, exist_ok=True)

    best_row = val_df.sort_values(by="mean_auc(t)", ascending=False).iloc[0]
    best_l1 = float(best_row["l1_ratio"])
    best_alpha = float(best_row["alpha"])

    t1 = time.time()
    risk_scores = get_heldout_risk_scores_CoxPH(
        event_pred_df,
        base_vars + type_cols,
        cfg["continuous_vars"],
        cfg["penalized_cols"],
        pca_config=cfg["pca_config"],
        event_col=args.event,
        tstop_col=f"tt_{args.event}",
        max_iter=args.max_iter,
        penalized=True,
        l1_ratio=best_l1,
        alpha=best_alpha,
        n_jobs=n_jobs,
        backend=args.backend,
    ).rename(columns={"risk_score": f"{args.modality}_risk_score"})
    risk_scores.to_csv(risk_fp, index=False)
    print(f"[time] {label} held-out risk: {(time.time() - t1) / 60:.1f}m ({len(risk_scores)} patients)")
    print(f"[done] {args.scheme}:{args.event}:{args.modality} -> {out_dir}")


if __name__ == "__main__":
    main()
