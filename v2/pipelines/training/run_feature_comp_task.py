"""Run Feature Comp Task script for model training workflows."""

import argparse
import json
import os
import time

import pandas as pd

from anchors import ANCHORS, DEFAULT_ANCHOR, age_col
from config import SURV_PATH
from schemes import SCHEMES, get_output_dir
from survival import get_heldout_risk_scores_CoxPH, run_grid_CoxPH_parallel

from pipelines.training.slurm_array_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    _get_n_jobs,
    filter_event_rows,
    get_events_from_df,
    load_feature_modalities_df,
    validate_cox_inputs,
)

SKIP_REPORT_DIR = os.path.join(SURV_PATH, "results", "skipped_events")


def _write_skip_report(scheme: str, event: str, run_type: str, reason: str) -> None:
    os.makedirs(SKIP_REPORT_DIR, exist_ok=True)
    report_fp = os.path.join(SKIP_REPORT_DIR, f"{scheme}_{run_type}_skipped.jsonl")
    entry = {"scheme": scheme, "event": event, "run_type": run_type, "reason": reason}
    with open(report_fp, "a") as f:
        f.write(json.dumps(entry) + "\n")


ALL_MODALITIES = ["stage", "treatment", "somatic", "prs", "text"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one feature-comparison modality for one endpoint event.")
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEMES.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--modality",
        required=True,
        choices=["stage", "treatment", "somatic", "prs", "text", "all"],
    )
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=sorted(ANCHORS.keys()))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def _run_one_modality(
    args: argparse.Namespace,
    modality: str,
    full_prediction_df: pd.DataFrame,
    type_cols_base: list[str],
    modality_cfg: dict,
) -> None:
    """Run grid search + held-out risk scores for one modality.

    Extracted from the original single-modality ``main()`` body verbatim (Phase-A A2): the
    per-modality try/except lives in the caller so one modality failing does not stop the rest
    when looping under ``--modality all``. All skip/overwrite logic and ``[time]``/``[done]``/
    ``[skip]`` print formats are unchanged.
    """
    if modality not in modality_cfg:
        raise ValueError(f"Unsupported modality '{modality}'.")

    out_dir = os.path.join(get_output_dir(args.scheme, "feature_comps", args.anchor), args.event)
    os.makedirs(out_dir, exist_ok=True)
    test_fp = os.path.join(out_dir, f"{modality}_test.csv")
    val_fp = os.path.join(out_dir, f"{modality}_val.csv")
    risk_dir = os.path.join(get_output_dir(args.scheme, "feature_comps", args.anchor), "..", "held_out_risk_scores", args.event)
    risk_dir = os.path.normpath(risk_dir)
    risk_fp = os.path.join(risk_dir, f"{modality}_risk_scores.csv")
    grid_done = os.path.exists(test_fp) and os.path.exists(val_fp)
    risk_done = os.path.exists(risk_fp)
    if (not args.overwrite) and grid_done and risk_done:
        print(f"[skip] Existing outputs found for {args.scheme}:{args.event}:{modality}")
        return

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.empty:
        reason = f"No rows with tt_{args.event} > 0"
        print(f"[skip-data] {args.scheme}:{args.event}:{modality} — {reason}")
        _write_skip_report(args.scheme, args.event, f"feature_comps_{modality}", reason)
        return

    base_vars = ["GENDER", age_col(args.anchor)]
    type_cols = type_cols_base
    cfg = modality_cfg[modality]
    all_feature_cols = base_vars + type_cols + cfg["continuous_vars"] + cfg["penalized_cols"]
    # deduplicate while preserving order
    seen = set()
    all_feature_cols = [c for c in all_feature_cols if not (c in seen or seen.add(c))]
    label = f"{args.scheme}:{args.event}:{modality}"
    try:
        event_pred_df, dropped_cols = validate_cox_inputs(
            event_pred_df, args.event, f"tt_{args.event}", all_feature_cols, label=label,
        )
    except ValueError as e:
        reason = str(e)
        print(f"[skip-data] {label} — {reason}")
        _write_skip_report(args.scheme, args.event, f"feature_comps_{modality}", reason)
        return
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
    ).rename(columns={"risk_score": f"{modality}_risk_score"})
    risk_scores.to_csv(risk_fp, index=False)
    print(f"[time] {label} held-out risk: {(time.time() - t1) / 60:.1f}m ({len(risk_scores)} patients)")
    print(f"[done] {args.scheme}:{args.event}:{modality} -> {out_dir}")


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    full_prediction_df, type_cols, _, modality_cfg, _ = load_feature_modalities_df(
        args.scheme, modality=args.modality, anchor=args.anchor
    )
    events = get_events_from_df(full_prediction_df)
    if args.event not in events:
        raise ValueError(
            f"Event '{args.event}' not found for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )

    if args.modality == "all":
        for modality in ALL_MODALITIES:
            try:
                _run_one_modality(args, modality, full_prediction_df, type_cols, modality_cfg)
            except Exception as e:
                print(f"[error] {args.scheme}:{args.event}:{modality} failed: {e}")
    else:
        _run_one_modality(args, args.modality, full_prediction_df, type_cols, modality_cfg)


if __name__ == "__main__":
    main()
