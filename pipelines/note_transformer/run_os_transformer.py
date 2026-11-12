"""Search / train / held-out test for the note-survival transformer (overall survival).

Stages
------
search  Random hyperparameter search selected on validation C-index only (test never touched).
train   One fit with the base config (no search).
test    Evaluate one selected checkpoint on the held-out test patients, then compare against the
        existing Cox text/base out-of-fold risk scores on identical patient pairs.
all     search -> test, passing the search result's checkpoint directly.

Inputs come from ``pipelines.note_transformer.prepare_os_inputs``. Outputs go to
``<death_met results dir>/note_transformer/<tag>/{search,single_run,test}/``.

Usage (from the repo root):

    python -m pipelines.note_transformer.run_os_transformer --stage all --num-trials 20
    python -m pipelines.note_transformer.run_os_transformer --stage test --checkpoint <best_model.pt>
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace

import numpy as np
import polars as pl
import torch

from anchors import ANCHORS, DEFAULT_ANCHOR
from schemes import full_cohort_event_dir, full_cohort_risk_dir, scheme_results_dir
from pipelines.note_transformer.prepare_os_inputs import (
    EVENT,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SCHEME,
    default_inputs_dir,
)
from pipelines.note_transformer.note_survival_transformer.config import Config
from pipelines.note_transformer.note_survival_transformer.dataset import load_data
from pipelines.note_transformer.note_survival_transformer.evaluate import safe_c_index
from pipelines.note_transformer.note_survival_transformer.hparam_search import run_search
from pipelines.note_transformer.note_survival_transformer.predict import final_test_evaluation
from pipelines.note_transformer.note_survival_transformer.train import fit

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "configs", "os_config.json")
DEFAULT_SEARCH_SPACE = os.path.join(HERE, "configs", "os_search_space.json")

COMPARISON_SCHEMA = {
    "evaluation_scope": pl.String,
    "landmark": pl.Float64,
    "model": pl.String,
    "n_patients": pl.Int64,
    "n_events": pl.Int64,
    "n_comparable_pairs": pl.Int64,
    "n_strata": pl.Int64,
    "n_evaluable_strata": pl.Int64,
    "c_index": pl.Float64,
    "c_index_reason": pl.String,
    "n_all_test": pl.Int64,
    "excluded_missing_score": pl.Int64,
    "excluded_nonfinite_score": pl.Int64,
    "excluded_invalid_fold": pl.Int64,
    "comparators_available": pl.String,
    "strata_definition": pl.String,
}
COX_COMPARATORS = {"cox_text_oof": ("text", "text_risk_score"), "cox_base_oof": ("base", "base_risk_score")}


# --------------------------------------------------------------------------- runtime helpers


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    return requested


def resolve_threads(requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--num-threads must be positive")
        return requested
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) >= 1:
        return int(slurm)
    return max(1, (os.cpu_count() or 2) - 1)


@contextlib.contextmanager
def nondeterministic_ok(enabled: bool):
    """Temporarily downgrade ``torch.use_deterministic_algorithms`` to warn-only.

    The vendored ``seed_everything`` forces strict determinism, which can raise on some CUDA
    kernels. Wrapping the function avoids editing vendored code; the previous state is restored.
    """
    if not enabled:
        yield
        return
    original = torch.use_deterministic_algorithms
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def relaxed(mode=True, *, warn_only=False):
        return original(mode, warn_only=True)

    torch.use_deterministic_algorithms = relaxed
    try:
        yield
    finally:
        torch.use_deterministic_algorithms = original
        original(prev_mode, warn_only=prev_warn)


def run_dirs(anchor: str, tag: str) -> dict[str, str]:
    root = os.path.join(scheme_results_dir(SCHEME, anchor), "note_transformer", tag)
    return {
        "root": root,
        "search": os.path.join(root, "search"),
        "single_run": os.path.join(root, "single_run"),
        "test": os.path.join(root, "test"),
    }


def load_inputs(inputs_dir: str, anchor: str):
    """Validate the preparation manifest and load the package data."""
    manifest_path = os.path.join(inputs_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run `python -m pipelines.note_transformer.prepare_os_inputs` first."
        )
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("signature", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest_path}: unsupported preparation schema; re-run prepare_os_inputs --overwrite")
    prepared_anchor = manifest["signature"].get("anchor")
    if prepared_anchor != anchor:
        raise ValueError(
            f"{inputs_dir}: prepared for anchor {prepared_anchor!r}, but --anchor is {anchor!r}. "
            "Use the matching --anchor or an input bundle prepared for the requested anchor."
        )
    missing = [f for f in manifest["expected_files"] if not os.path.exists(os.path.join(inputs_dir, f))]
    if missing:
        raise FileNotFoundError(f"{inputs_dir} is missing {missing}")
    static = bool(manifest["static"]["enabled"])
    has_patients = os.path.exists(os.path.join(inputs_dir, "patients.csv"))
    if static != has_patients:
        raise ValueError(f"{inputs_dir}: manifest static={static} but patients.csv present={has_patients}")
    data = load_data(
        os.path.join(inputs_dir, "notes.npy"),
        os.path.join(inputs_dir, "notes.csv"),
        outcomes_path=os.path.join(inputs_dir, "outcomes.csv"),
        splits_path=os.path.join(inputs_dir, "splits.csv"),
        patients_path=os.path.join(inputs_dir, "patients.csv") if static else None,
        embedding_model_id=manifest["embedding_model_id"],
    )
    return data, manifest


def build_config(config_path, landmarks, device, num_threads, num_epochs=None) -> Config:
    config = Config.from_json(config_path)
    changes = {"device": device, "num_threads": num_threads}
    if landmarks is not None:
        changes["landmarks"] = tuple(float(x) for x in landmarks)
    if num_epochs is not None:
        changes["num_epochs"] = num_epochs
        changes["min_epochs"] = min(config.min_epochs, num_epochs)
    return replace(config, **changes).validate()


def prepare_output_dir(path: str, overwrite: bool, resume: bool) -> str:
    if os.path.isdir(path) and os.listdir(path):
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(
                f"{path} already exists and is not empty. Use --resume to continue it, "
                "--overwrite to replace it, or a different --tag."
            )
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- stages


def stage_search(args, data, config, dirs) -> str:
    with open(args.search_space) as handle:
        space = json.load(handle)
    out = prepare_output_dir(dirs["search"], args.overwrite, args.resume)
    with nondeterministic_ok(args.allow_nondeterministic):
        summary = run_search(data, config, space, args.num_trials, args.search_seed, out, resume=args.resume)
    print(
        f"[done] search: trial {summary['selected_trial']} selected, validation C-index "
        f"{summary['validation_score']:.4f}; checkpoint {summary['checkpoint_path']}"
    )
    return summary["checkpoint_path"]


def stage_train(args, data, config, dirs) -> str:
    out = prepare_output_dir(dirs["single_run"], args.overwrite, args.resume)
    with nondeterministic_ok(args.allow_nondeterministic):
        result = fit(data, config, out, resume=args.resume)
    print(f"[done] train: validation C-index {result['validation_score']:.4f}; checkpoint {result['checkpoint_path']}")
    return result["checkpoint_path"]


def pick_checkpoint(explicit: str | None, dirs: dict[str, str]) -> str:
    """Resolve the checkpoint for a standalone test stage."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--checkpoint {explicit} does not exist")
        return os.path.abspath(explicit)
    candidates = [
        os.path.join(dirs["search"], "best_model.pt"),
        os.path.join(dirs["single_run"], "best_model.pt"),
    ]
    existing = [p for p in candidates if os.path.exists(p)]
    if len(existing) == 2:
        raise ValueError(
            "Both candidate checkpoints exist; pass --checkpoint to choose one:\n  "
            + "\n  ".join(existing)
        )
    if not existing:
        raise FileNotFoundError(
            "No checkpoint found. Looked for:\n  " + "\n  ".join(candidates)
            + "\nRun --stage search or --stage train first, or pass --checkpoint."
        )
    return os.path.abspath(existing[0])


def stage_test(args, data, checkpoint: str, dirs, device: str, num_threads: int) -> str:
    out = prepare_output_dir(dirs["test"], args.overwrite, args.resume)
    print(f"[test] checkpoint: {checkpoint}")
    with nondeterministic_ok(args.allow_nondeterministic):
        report = final_test_evaluation(checkpoint, data, out, device=device, num_threads=num_threads)
    print(f"[test] model_id={report['model_id']} weighted test C-index={report['weighted_test_c_index']:.4f}")
    comparison = build_comparison(
        os.path.join(out, "test_predictions.csv"),
        args.anchor,
        os.path.join(out, "comparison.csv"),
        test_report=report,
    )
    compact = ["evaluation_scope", "landmark", "model", "n_patients", "n_events", "n_comparable_pairs",
               "c_index", "c_index_reason"]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=40, tbl_width_chars=200):
        print(comparison.select([c for c in compact if c in comparison.columns]))
    print(f"[note] full table with exclusion counts: {os.path.join(out, 'comparison.csv')}")
    return out


# --------------------------------------------------------------------------- comparison


def admissible_pairs(time, event) -> int:
    """Comparable pairs under Harrell's rule: sum_t events_at_t * (n_later_t + censored_at_t).

    Same-time event/censor pairs are comparable; same-time event/event pairs are not.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=np.int64)
    if t.size == 0:
        return 0
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    _, start, counts = np.unique(t, return_index=True, return_counts=True)
    events_at = np.add.reduceat(e, start)
    censored_at = counts - events_at
    later = t.size - (start + counts)
    return int(np.sum(events_at * (later + censored_at)))


def _read_cox_scores(path: str, score_col: str, name: str):
    """Return ``(frame, reason)``; frame is None when the comparator is unusable."""
    if not os.path.exists(path):
        print(f"[note] {path} not found; {name} comparison unavailable")
        return None, "missing_cox_file"
    try:
        df = pl.read_csv(path)
        if not {"DFCI_MRN", "outer_fold", score_col} <= set(df.columns):
            raise ValueError("missing required columns")
        df = df.select(
            pl.col("DFCI_MRN").cast(pl.Int64, strict=False).cast(pl.String).alias("patient_id"),
            pl.col("outer_fold").cast(pl.Float64, strict=False).alias(f"{name}_fold"),
            pl.col(score_col).cast(pl.Float64, strict=False).alias(f"{name}_score"),
        )
        if df["patient_id"].null_count() or df["patient_id"].n_unique() != df.height:
            raise ValueError("null or duplicate patient IDs")
    except Exception as exc:  # unreadable file, wrong schema, duplicates
        print(f"[note] {path} unusable ({exc}); {name} comparison unavailable")
        return None, "invalid_cox_file"
    return df, None


def _null_row(**kw) -> dict:
    row = {k: None for k in COMPARISON_SCHEMA}
    row.update(kw)
    return row


def _stratified_c_index(frame: pl.DataFrame, score_col: str, strata_col: str | None) -> dict:
    """Pair-weighted within-stratum Harrell C-index on one retained patient set."""
    out = {"n_patients": frame.height, "n_events": int(frame["event"].sum()) if frame.height else 0,
           "n_comparable_pairs": 0, "n_strata": 0, "n_evaluable_strata": 0, "c_index": None}
    if frame.height == 0:
        out["c_index_reason"] = "empty_intersection"
        return out
    if frame.height < 2:
        out["c_index_reason"] = "fewer_than_two_patients"
        return out
    if out["n_events"] == 0:
        out["c_index_reason"] = "no_primary_events"
        return out
    groups = [frame] if strata_col is None else frame.partition_by(strata_col, maintain_order=True)
    out["n_strata"] = len(groups)
    num = den = 0.0
    for group in groups:
        time = group["time_from_landmark"].to_numpy()
        event = group["event"].to_numpy()
        pairs = admissible_pairs(time, event)
        if pairs == 0:
            continue
        c, _ = safe_c_index(time, group[score_col].to_numpy(), event)
        if not np.isfinite(c):
            continue
        num += c * pairs
        den += pairs
        out["n_evaluable_strata"] += 1
    out["n_comparable_pairs"] = int(den)
    if den == 0:
        out["c_index_reason"] = "no_comparable_pairs"
    else:
        out["c_index"], out["c_index_reason"] = num / den, None
    return out


def _all_test_rows(preds: pl.DataFrame, test_report: dict | None) -> list[dict]:
    rows = []
    if test_report is not None:
        metrics = test_report["landmark_metrics"]
        score = test_report.get("weighted_test_c_index")
    else:
        metrics, total_c, total_n = [], 0.0, 0
        for landmark in sorted(preds["landmark"].unique().to_list()):
            f = preds.filter(pl.col("landmark") == landmark)
            c, reason = safe_c_index(f["time_from_landmark"].to_numpy(), f["log_risk_event"].to_numpy(), f["event"].to_numpy())
            metrics.append({"landmark": landmark, "c_index": c, "c_index_reason": reason,
                            "n_eligible": f.height, "n_events": int(f["event"].sum())})
            if np.isfinite(c):
                total_c += c * f.height
                total_n += f.height
        score = total_c / total_n if total_n else float("nan")
    for m in metrics:
        c = m["c_index"]
        rows.append(_null_row(
            evaluation_scope="all_test", landmark=float(m["landmark"]), model="transformer",
            n_patients=int(m["n_eligible"]), n_events=int(m["n_events"]),
            n_all_test=int(m["n_eligible"]),
            c_index=float(c) if c is not None and np.isfinite(c) else None,
            c_index_reason=m.get("c_index_reason"),
        ))
    rows.append(_null_row(
        evaluation_scope="all_test", landmark=None, model="transformer_weighted_by_eligible",
        n_patients=int(sum(m["n_eligible"] for m in metrics)), n_events=int(sum(m["n_events"] for m in metrics)),
        c_index=float(score) if score is not None and np.isfinite(score) else None,
        c_index_reason=None if score is not None and np.isfinite(score) else "no_evaluable_landmark",
    ))
    return rows


def _reported_cox_rows(cox_test_dir: str) -> list[dict]:
    rows = []
    for model, fname in (("cox_text_test", "text_test.csv"), ("cox_base_test", "base_test.csv")):
        path = os.path.join(cox_test_dir, fname)
        if not os.path.exists(path):
            rows.append(_null_row(evaluation_scope="reported_cox_holdout", landmark=0.0, model=model,
                                  c_index_reason="missing_cox_file"))
            continue
        try:
            df = pl.read_csv(path)
            value = float(df["mean_c_index"][0])
            reason = None if np.isfinite(value) else "nonfinite_reported_value"
            value = value if np.isfinite(value) else None
        except Exception:
            rows.append(_null_row(evaluation_scope="reported_cox_holdout", landmark=0.0, model=model,
                                  c_index_reason="invalid_cox_file"))
            continue
        rows.append(_null_row(evaluation_scope="reported_cox_holdout", landmark=0.0, model=model,
                              c_index=value, c_index_reason=reason))
    return rows


def build_comparison(
    test_predictions_csv: str,
    anchor: str,
    out_csv: str | None = None,
    *,
    risk_dir: str | None = None,
    cox_test_dir: str | None = None,
    test_report: dict | None = None,
) -> pl.DataFrame:
    """Compare the transformer to the Cox OOF scores on identical within-fold patient pairs.

    Cox ``*_risk_scores.csv`` are out-of-fold with independently fitted folds, so raw scores are
    never ranked across folds. Patients are stratified by their available Cox fold IDs and only
    within-stratum pairs are compared, with one common pair set for every model. Per-stratum
    C-indices are averaged weighted by admissible pairs. This is descriptive: the transformer and
    Cox models were trained on different partitions.
    """
    risk_dir = risk_dir or full_cohort_risk_dir(SCHEME, EVENT, anchor)
    cox_test_dir = cox_test_dir or full_cohort_event_dir(SCHEME, EVENT, anchor)
    # Header-only predictions are valid when no test patient survives any landmark.
    preds = pl.read_csv(test_predictions_csv, schema_overrides={
        "patient_id": pl.String, "landmark": pl.Float64, "log_risk_event": pl.Float64,
        "time_from_landmark": pl.Float64, "event": pl.Int64,
    })
    comparators, reasons = {}, {}
    for model, (prefix, score_col) in COX_COMPARATORS.items():
        frame, reason = _read_cox_scores(os.path.join(risk_dir, f"{prefix}_risk_scores.csv"), score_col, prefix)
        if frame is not None:
            comparators[model] = (prefix, frame)
        else:
            reasons[model] = reason
    available = [m for m in COX_COMPARATORS if m in comparators]
    strata_def = "|".join(f"{COX_COMPARATORS[m][0]}_fold" for m in available) or None

    rows = _all_test_rows(preds, test_report)
    landmarks = set(preds["landmark"].unique().to_list())
    if test_report is not None:
        landmarks.update(float(m["landmark"]) for m in test_report["landmark_metrics"])
    for landmark in sorted(landmarks):
        frame = preds.filter(pl.col("landmark") == landmark).select(
            "patient_id", "log_risk_event", "time_from_landmark", "event"
        )
        n_all = frame.height
        for model, (prefix, cox) in comparators.items():
            frame = frame.join(cox, on="patient_id", how="left")
        counts = {"excluded_missing_score": 0, "excluded_nonfinite_score": 0, "excluded_invalid_fold": 0}
        score_cols = [f"{COX_COMPARATORS[m][0]}_score" for m in available]
        fold_cols = [f"{COX_COMPARATORS[m][0]}_fold" for m in available]
        checks = (
            ("excluded_missing_score", [pl.col(c).is_not_null() for c in score_cols]),
            ("excluded_nonfinite_score", [pl.col(c).is_finite() for c in score_cols]),
            ("excluded_invalid_fold", [
                pl.col(c).is_finite() & (pl.col(c) >= 0) & (pl.col(c) == pl.col(c).floor()) for c in fold_cols
            ]),
        )
        for key, conds in checks:
            if not conds:
                continue
            kept = frame.filter(pl.all_horizontal([c.fill_null(False) for c in conds]))
            counts[key] = frame.height - kept.height
            frame = kept
        strata_col = None
        if available:
            frame = frame.with_columns(
                pl.concat_str([pl.col(c).cast(pl.Int64).cast(pl.String) for c in fold_cols], separator="|").alias("_stratum")
            )
            strata_col = "_stratum"
        base_audit = dict(
            evaluation_scope="within_cox_folds", landmark=float(landmark), n_all_test=n_all,
            comparators_available=",".join(available) or None, strata_definition=strata_def, **counts,
        )
        models = {"transformer": "log_risk_event"}
        models.update({m: f"{COX_COMPARATORS[m][0]}_score" for m in available})
        for model, column in models.items():
            if not available:
                rows.append(_null_row(model=model, c_index_reason="no_cox_comparator_available",
                                      n_patients=0, n_events=0, **base_audit))
                continue
            stats = _stratified_c_index(frame, column, strata_col)
            rows.append(_null_row(model=model, **stats, **base_audit))
        for model, reason in reasons.items():
            rows.append(_null_row(model=model, c_index_reason=reason, **base_audit))
    rows += _reported_cox_rows(cox_test_dir)
    out = pl.from_dicts(rows, schema=COMPARISON_SCHEMA)
    if out_csv:
        out.write_csv(out_csv)
        print(f"[done] comparison written to {out_csv}")
    return out


# --------------------------------------------------------------------------- CLI


def _parse_landmarks(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=sorted(ANCHORS))
    p.add_argument("--stage", default="all", choices=["search", "train", "test", "all"])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--search-space", default=DEFAULT_SEARCH_SPACE)
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--search-seed", type=int, default=42)
    p.add_argument("--landmarks", type=_parse_landmarks, default=None,
                   help="comma-separated days; default: the config file's landmarks (0,90,180,365)")
    p.add_argument("--n-epochs", type=int, default=None, help="override num_epochs (smoke runs)")
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--inputs-dir", default=None)
    p.add_argument("--tag", default="default")
    p.add_argument("--checkpoint", default=None, help="explicit checkpoint; only valid with --stage test")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-nondeterministic", action="store_true")
    args = p.parse_args(argv)
    if args.checkpoint and args.stage != "test":
        p.error("--checkpoint is only valid with --stage test")
    if args.resume and args.overwrite:
        p.error("--resume and --overwrite are mutually exclusive")
    if args.num_trials < 1:
        p.error("--num-trials must be at least 1")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    device = resolve_device(args.device)
    threads = resolve_threads(args.num_threads)
    dirs = run_dirs(args.anchor, args.tag)
    inputs_dir = os.path.abspath(args.inputs_dir or default_inputs_dir(args.anchor, "text", None))
    print(f"[run] stage={args.stage} anchor={args.anchor} device={device} threads={threads}")
    print(f"[run] inputs={inputs_dir}\n[run] outputs={dirs['root']}")
    t0 = time.time()
    data, manifest = load_inputs(inputs_dir, args.anchor)
    print(f"[run] split_check={manifest['split_check']}; patients={manifest['counts']['patients']:,}")

    checkpoint = None
    if args.stage in ("search", "train", "all"):
        config = build_config(args.config, args.landmarks, device, threads, args.n_epochs)
        if args.stage == "train":
            checkpoint = stage_train(args, data, config, dirs)
        else:
            checkpoint = stage_search(args, data, config, dirs)
    if args.stage == "test":
        checkpoint = pick_checkpoint(args.checkpoint, dirs)
    if args.stage in ("test", "all"):
        stage_test(args, data, checkpoint, dirs, device, threads)
    print(f"[done] {args.stage} finished in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
