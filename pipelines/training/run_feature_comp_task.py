"""Run Feature Comp Task script for model training workflows."""

import argparse
import copy
import json
import os
import time

import polars as pl

from anchors import ANCHORS, DEFAULT_ANCHOR, age_col
from config import SURV_PATH
from schemes import SCHEMES, get_output_dir
from shared.icd10 import MET_SITE_GROUPS
from survival import get_nested_heldout_risk_scores_CoxPH, run_grid_CoxPH_parallel

from pipelines.training.slurm_array_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    adaptive_low_alphas_for,
    _get_n_jobs,
    filter_event_rows,
    get_events_from_df,
    load_feature_modalities_df,
    parse_float_list,
    validate_cox_inputs,
    write_ipcw_reference_csv,
)

SKIP_REPORT_DIR = os.path.join(SURV_PATH, "results", "skipped_events")


def _write_skip_report(scheme: str, event: str, run_type: str, reason: str) -> None:
    os.makedirs(SKIP_REPORT_DIR, exist_ok=True)
    report_fp = os.path.join(SKIP_REPORT_DIR, f"{scheme}_{run_type}_skipped.jsonl")
    entry = {"scheme": scheme, "event": event, "run_type": run_type, "reason": reason}
    with open(report_fp, "a") as f:
        f.write(json.dumps(entry) + "\n")


ALL_MODALITIES = ["stage", "treatment", "somatic", "prs", "text", "metburden"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one feature-comparison modality for one endpoint event.")
    parser.add_argument("--scheme", required=True, choices=sorted(SCHEMES.keys()))
    # --event and --events are mutually exclusive but jointly required, so every existing
    # caller (--event <one>) keeps working verbatim; --events is purely additive.
    event_group = parser.add_mutually_exclusive_group(required=True)
    event_group.add_argument("--event")
    event_group.add_argument(
        "--events",
        nargs="+",
        metavar="EVENT",
        help=(
            "Run several endpoints in ONE process, reusing a single load of the feature "
            "frame. load_feature_modalities_df does not depend on the event, so batching "
            "here avoids re-reading the embedding/cancer-type/modality files and "
            "recomputing the common-MRN intersection once per event. Each event is "
            "isolated by try/except, exactly as --modality all isolates each modality: "
            "one failing event does not stop the rest, and the process still exits "
            "non-zero if any failed."
        ),
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["stage", "treatment", "somatic", "prs", "text", "metburden", "all"],
    )
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=sorted(ANCHORS.keys()))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alphas", type=parse_float_list, default=DEFAULT_ALPHAS)
    parser.add_argument("--l1-ratios", type=parse_float_list, default=DEFAULT_L1_RATIOS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky"])
    return parser.parse_args()


def _run_one_modality(
    args: argparse.Namespace,
    modality: str,
    full_prediction_df: pl.DataFrame,
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
    ipcw_fp = os.path.join(out_dir, f"{modality}_ipcw_reference.csv.gz")
    risk_dir = os.path.join(get_output_dir(args.scheme, "feature_comps", args.anchor), "..", "held_out_risk_scores", args.event)
    risk_dir = os.path.normpath(risk_dir)
    risk_fp = os.path.join(risk_dir, f"{modality}_risk_scores.csv")
    grid_done = os.path.exists(test_fp) and os.path.exists(val_fp) and os.path.exists(ipcw_fp)
    risk_done = os.path.exists(risk_fp)
    if (not args.overwrite) and grid_done and risk_done:
        print(f"[skip] Existing outputs found for {args.scheme}:{args.event}:{modality}")
        return

    event_pred_df = filter_event_rows(full_prediction_df, args.event)
    if event_pred_df.is_empty():
        reason = f"No rows with tt_{args.event} > 0"
        print(f"[skip-data] {args.scheme}:{args.event}:{modality} — {reason}")
        _write_skip_report(args.scheme, args.event, f"feature_comps_{modality}", reason)
        raise RuntimeError(reason)

    base_vars = ["GENDER", age_col(args.anchor)]
    type_cols = type_cols_base
    # Deep-copy: modality_cfg is built once per process and shared across every
    # modality in the ``--modality all`` in-process loop (see main() below).
    # The constant-column drop a few lines down, and the concordant-site drop
    # for metburden, both mutate cfg's lists in place — without copying first,
    # a column dropped while processing one modality would stay dropped for
    # every subsequent modality in the same loop iteration's modality_cfg.
    cfg = copy.deepcopy(modality_cfg[modality])
    if modality == "metburden" and args.event.endswith("M") and args.event[:-1] in MET_SITE_GROUPS:
        # LEAKAGE GUARD: MET_SITE_{site} and the `{site}M` outcome describe the
        # same anatomy from different sources (pre-index ICD vs. post-index
        # clinical extraction) — see build_met_burden_df's LEAKAGE WARNING.
        # Drop only the concordant site's indicator; N_MET_SITES (which
        # aggregates all eight sites) is kept since the concordant site
        # contributes at most 1 to it.
        concordant_col = f"MET_SITE_{args.event[:-1]}"
        cfg["penalized_cols"] = [c for c in cfg["penalized_cols"] if c != concordant_col]
        # The aggregate count also contained the concordant anatomy.  Replace
        # it with a count of other metastatic sites so no version of the
        # outcome site's baseline proxy remains in the design matrix.
        event_pred_df = event_pred_df.with_columns(
            (pl.col("N_MET_SITES") - pl.col(concordant_col))
            .clip(lower_bound=0)
            .alias("N_OTHER_MET_SITES")
        )
        cfg["continuous_vars"] = [
            "N_OTHER_MET_SITES" if c == "N_MET_SITES" else c
            for c in cfg["continuous_vars"]
        ]
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
        raise RuntimeError(reason) from e
    type_cols = [c for c in type_cols if c not in dropped_cols]
    cfg["continuous_vars"] = [c for c in cfg["continuous_vars"] if c not in dropped_cols]
    cfg["penalized_cols"] = [c for c in cfg["penalized_cols"] if c not in dropped_cols]
    all_feature_cols = [c for c in all_feature_cols if c not in dropped_cols]
    n_before = len(event_pred_df)
    # Columns with a paired _missing indicator are raw measurements deferred to per-fold
    # mean imputation (e.g. lab values). All other feature NaNs are unexpected and drop the row.
    imputable_cols = {c for c in all_feature_cols if f"{c}_missing" in event_pred_df.columns}
    drop_subset = [c for c in all_feature_cols if c not in imputable_cols] + [args.event, f"tt_{args.event}"]
    event_pred_df = event_pred_df.filter(
        pl.all_horizontal([
            pl.col(c).cast(pl.Float64, strict=False).is_finite()
            for c in drop_subset
        ])
    )
    n_dropped = n_before - len(event_pred_df)
    if n_dropped > 0:
        print(f"{label} Dropped {n_dropped}/{n_before} rows with NaN values")
    # For small feature sets the joblib overhead dominates; run serially.
    n_jobs = 1 if len(cfg["penalized_cols"]) < 50 else _get_n_jobs(args.n_jobs)

    # --- Grid search (skip if already done, reuse val results for risk scores) ---
    if (not args.overwrite) and grid_done:
        print(f"[skip] {label} grid search already done, loading val results")
        val_df = pl.read_csv(val_fp)
    else:
        t0 = time.time()
        test_df, val_df, _, ipcw_df = run_grid_CoxPH_parallel(
            event_pred_df,
            base_vars + type_cols,
            cfg["continuous_vars"],
            cfg["penalized_cols"],
            args.l1_ratios,
            args.alphas,
            pca_config=cfg["pca_config"],
            event_col=args.event,
            tstop_col=f"tt_{args.event}",
            max_iter=args.max_iter,
            n_splits=args.n_splits,
            n_jobs=n_jobs,
            backend=args.backend,
            return_audit=True,
            adaptive_low_alphas=adaptive_low_alphas_for(args.alphas),
        )
        test_df.write_csv(test_fp)
        val_df.write_csv(val_fp)
        write_ipcw_reference_csv(
            ipcw_df, ipcw_fp
        )
        print(f"[time] {label} grid search: {(time.time() - t0) / 60:.1f}m")

    # --- Generate nested held-out risk scores (tuning inside each outer fold) ---
    os.makedirs(risk_dir, exist_ok=True)

    t1 = time.time()
    risk_scores = get_nested_heldout_risk_scores_CoxPH(
        event_pred_df,
        base_vars + type_cols,
        cfg["continuous_vars"],
        cfg["penalized_cols"],
        args.l1_ratios,
        args.alphas,
        pca_config=cfg["pca_config"],
        event_col=args.event,
        tstop_col=f"tt_{args.event}",
        max_iter=args.max_iter,
        n_splits=args.n_splits,
        n_jobs=n_jobs,
        backend=args.backend,
        adaptive_low_alphas=adaptive_low_alphas_for(args.alphas),
    ).rename({"risk_score": f"{modality}_risk_score"})
    risk_scores.write_csv(risk_fp)
    print(f"[time] {label} held-out risk: {(time.time() - t1) / 60:.1f}m ({len(risk_scores)} patients)")
    print(f"[done] {args.scheme}:{args.event}:{modality} -> {out_dir}")


def _run_all_modalities_for_event(
    args: argparse.Namespace,
    full_prediction_df: pl.DataFrame,
    type_cols: list[str],
    modality_cfg: dict,
) -> list[str]:
    """Run the requested modality (or every modality) for ``args.event``.

    Returns the list of modality names that failed; the caller decides whether a failure
    is fatal. Extracted from main() unchanged so that the per-event batching loop and the
    original single-event path share exactly one code path.
    """
    if args.modality == "all":
        failures = []
        for modality in ALL_MODALITIES:
            try:
                _run_one_modality(args, modality, full_prediction_df, type_cols, modality_cfg)
            except Exception as e:
                print(f"[error] {args.scheme}:{args.event}:{modality} failed: {e}")
                failures.append(modality)
        return failures
    _run_one_modality(args, args.modality, full_prediction_df, type_cols, modality_cfg)
    return []


def main() -> None:
    args = _parse_args()
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    # --events is the batched form; --event is the original single-endpoint form. Both
    # normalize to a list here so there is one downstream code path. argparse guarantees
    # exactly one of the two is set.
    requested_events = args.events if args.events else [args.event]

    # Loaded ONCE for every event in the batch: load_feature_modalities_df reads the
    # embedding frame, cancer-type file and modality feature files and computes the
    # common-MRN intersection, none of which depend on the event.
    full_prediction_df, type_cols, _, modality_cfg, _ = load_feature_modalities_df(
        args.scheme, modality=args.modality, anchor=args.anchor
    )
    events = get_events_from_df(full_prediction_df)

    missing = [e for e in requested_events if e not in events]
    if missing:
        # Single-event form keeps its original message and fails fast. Batched form skips
        # the unknown endpoints and still runs the rest, then reports them at the end.
        if not args.events:
            raise ValueError(
                f"Event '{args.event}' not found for scheme '{args.scheme}'. "
                f"Found {len(events)} events."
            )
        for event in missing:
            print(f"[skip-data] {args.scheme}:{event} — not found for this scheme")

    runnable = [e for e in requested_events if e in events]
    if not runnable:
        raise ValueError(
            f"None of the requested events exist for scheme '{args.scheme}'. "
            f"Found {len(events)} events."
        )

    if not args.events:
        # Original single-event behaviour, byte-for-byte: failures propagate as before.
        failures = _run_all_modalities_for_event(args, full_prediction_df, type_cols, modality_cfg)
        if failures:
            raise RuntimeError(f"Failed modalities: {', '.join(failures)}")
        return

    # Batched form: isolate each event so one bad endpoint does not lose the whole
    # process's remaining work (the array script's rows-per-task is sized on the
    # assumption that a task keeps going).
    failed_events: list[str] = []
    for i, event in enumerate(runnable, start=1):
        # Per-event shallow copy: _run_one_modality reads args.event throughout, and
        # mutating the shared namespace in place would leak state between iterations.
        event_args = copy.copy(args)
        event_args.event = event
        print(f"[event {i}/{len(runnable)}] {args.scheme}:{event}")
        try:
            failures = _run_all_modalities_for_event(
                event_args, full_prediction_df, type_cols, modality_cfg
            )
            if failures:
                failed_events.append(event)
        except Exception as e:
            print(f"[error] {args.scheme}:{event} failed: {e}")
            failed_events.append(event)

    if failed_events or missing:
        parts = []
        if failed_events:
            parts.append(f"failed events: {', '.join(failed_events)}")
        if missing:
            parts.append(f"unknown events: {', '.join(missing)}")
        raise RuntimeError("; ".join(parts))


if __name__ == "__main__":
    main()
