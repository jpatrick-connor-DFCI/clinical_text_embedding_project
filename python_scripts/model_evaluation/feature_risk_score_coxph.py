"""Summarize second-stage CoxPH models over held-out modality risk scores."""

import argparse
import json
import os
import sys
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm

from embed_surv_utils import evaluate_surv_model

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent / "model_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from slurm_array_utils import SCHEME_CONFIG, get_events_from_df, get_output_dir, load_embedding_prediction_df

MODALITY_CHOICES = ["stage", "treatment", "labs", "somatic", "prs", "text"]
DEFAULT_MODALITIES = ["text", "stage", "treatment", "somatic", "prs"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit univariate and joint CoxPH models on held-out modality risk scores "
            "for events that have all requested modality risk-score files."
        )
    )
    parser.add_argument(
        "--scheme",
        nargs="+",
        choices=sorted(SCHEME_CONFIG),
        default=None,
        help="One or more schemes to analyze.",
    )
    parser.add_argument(
        "--all-schemes",
        action="store_true",
        help="Run across all schemes in SCHEME_CONFIG.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=DEFAULT_MODALITIES,
        choices=MODALITY_CHOICES,
        help="Modalities whose held-out risk scores must be present for an event to be analyzed.",
    )
    parser.add_argument(
        "--event",
        dest="events",
        action="append",
        default=None,
        help="Optional event to analyze. Repeat --event to run a subset.",
    )
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to <scheme results>/risk_score_coxph.",
    )
    args = parser.parse_args()
    if args.all_schemes and args.scheme:
        parser.error("Use either --scheme or --all-schemes, not both.")
    if not args.all_schemes and not args.scheme:
        parser.error("Specify --scheme <name> [<name> ...] or --all-schemes.")
    return args


def _make_surv_array(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    y = np.empty(len(event), dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
    y["Status"] = np.asarray(event).astype(bool)
    y["Survival_in_days"] = np.asarray(time, dtype=np.float64)
    return y


def _default_output_dir(scheme: str) -> str:
    feature_comp_dir = Path(get_output_dir(scheme, "feature_comps"))
    return str(feature_comp_dir.parent / "risk_score_coxph")


def _resolve_output_dir(base_output_dir: str | None, scheme: str, multi_scheme: bool) -> str:
    if base_output_dir is None:
        return _default_output_dir(scheme)
    if multi_scheme:
        return os.path.join(base_output_dir, scheme)
    return base_output_dir


def _write_skip_report(skip_fp: str, scheme: str, event: str, reason: str) -> None:
    entry = {"scheme": scheme, "event": event, "reason": reason}
    with open(skip_fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _normalize_modalities(modalities: list[str]) -> list[str]:
    seen: set[str] = set()
    return [m for m in modalities if not (m in seen or seen.add(m))]


def _load_event_risk_profile(
    risk_root: str,
    event: str,
    modalities: list[str],
) -> tuple[pd.DataFrame | None, str | None]:
    risk_dfs: list[pd.DataFrame] = []
    missing_modalities: list[str] = []

    for modality in modalities:
        risk_fp = os.path.join(risk_root, event, f"{modality}_risk_scores.csv.gz")
        if not os.path.exists(risk_fp):
            missing_modalities.append(modality)
            continue

        cur = pd.read_csv(risk_fp)
        risk_col = f"{modality}_risk_score"
        if risk_col not in cur.columns:
            if "risk_score" in cur.columns:
                cur = cur.rename(columns={"risk_score": risk_col})
            else:
                missing_modalities.append(modality)
                continue

        cols = ["DFCI_MRN", risk_col]
        cur = cur[cols].dropna(subset=["DFCI_MRN"]).drop_duplicates(subset=["DFCI_MRN"])
        risk_dfs.append(cur)

    if missing_modalities:
        return None, f"Missing risk-score files/columns for modalities: {sorted(missing_modalities)}"
    if not risk_dfs:
        return None, "No risk-score files loaded"

    merged = reduce(lambda left, right: left.merge(right, on="DFCI_MRN", how="inner"), risk_dfs)
    if merged.empty:
        return None, "Risk-score inner merge produced 0 rows"
    return merged, None


def _fit_risk_score_coxph(
    df: pd.DataFrame,
    feature_cols: list[str],
    event_col: str,
    tstop_col: str,
    max_iter: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    model_df = df[[event_col, tstop_col] + feature_cols].dropna().copy()
    model_df = model_df.loc[model_df[tstop_col] > 0].copy()

    if model_df.empty:
        raise ValueError("No rows available after dropping NaN/non-positive survival rows")

    n_events = int(model_df[event_col].sum())
    n_censored = int(len(model_df) - n_events)
    if n_events == 0 or n_censored == 0:
        raise ValueError(f"Need both events and censored observations, got events={n_events}, censored={n_censored}")

    constant_cols = [c for c in feature_cols if model_df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        raise ValueError(f"Constant risk-score columns: {constant_cols}")

    X = model_df[feature_cols].astype(float)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols,
        index=model_df.index,
    )
    y = _make_surv_array(model_df[event_col].to_numpy(), model_df[tstop_col].to_numpy())

    lower, upper = np.percentile(model_df[tstop_col], [5, 95])
    eval_times = np.linspace(lower, upper, 50) if lower != upper else np.array([lower], dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = CoxPHSurvivalAnalysis(n_iter=max_iter)
        model.fit(X_scaled, y)
        mean_auc_t, ibs, c_index = evaluate_surv_model(model, X_scaled, y, y, eval_times)

    metrics = {
        "n": int(len(model_df)),
        "n_events": n_events,
        "n_censored": n_censored,
        "mean_c_index": float(c_index),
        "mean_auc(t)": float(mean_auc_t),
        "mean_ibs": float(ibs),
    }
    coef_df = pd.DataFrame({"covariate": feature_cols, "beta": model.coef_})
    return metrics, coef_df


def _run_one_scheme(args: argparse.Namespace, scheme: str, multi_scheme: bool) -> dict[str, object]:
    out_dir = _resolve_output_dir(args.output_dir, scheme, multi_scheme=multi_scheme)
    os.makedirs(out_dir, exist_ok=True)

    univar_fp = os.path.join(out_dir, "univariate_modality_metrics.csv.gz")
    joint_fp = os.path.join(out_dir, "joint_model_metrics.csv.gz")
    beta_fp = os.path.join(out_dir, "joint_model_betas.csv.gz")
    skip_fp = os.path.join(out_dir, "skipped_events.jsonl")
    meta_fp = os.path.join(out_dir, "metadata.json")

    if (
        (not args.overwrite)
        and os.path.exists(univar_fp)
        and os.path.exists(joint_fp)
        and os.path.exists(beta_fp)
    ):
        print(f"[skip] Existing outputs found for {scheme} in {out_dir}")
        return {"scheme": scheme, "status": "skipped_existing", "output_dir": out_dir}

    for fp in (univar_fp, joint_fp, beta_fp, skip_fp, meta_fp):
        if os.path.exists(fp):
            os.remove(fp)

    tte_df = load_embedding_prediction_df(scheme)
    available_events = set(get_events_from_df(tte_df))
    events = args.events or sorted(available_events)
    unknown_events = [event for event in events if event not in available_events]
    if unknown_events:
        raise ValueError(f"Unknown events for scheme '{scheme}': {unknown_events}")

    risk_root = os.path.normpath(os.path.join(get_output_dir(scheme, "feature_comps"), "..", "held_out_risk_scores"))

    univariate_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []

    tte_keep_cols = ["DFCI_MRN"]
    for event in events:
        tte_keep_cols.extend([event, f"tt_{event}"])
    tte_sub = tte_df[sorted(set(tte_keep_cols), key=tte_keep_cols.index)].copy()

    for event in tqdm(events):
        risk_profile, error = _load_event_risk_profile(risk_root, event, args.modalities)
        if error is not None:
            _write_skip_report(skip_fp, scheme, event, error)
            continue

        event_df = (
            risk_profile.merge(tte_sub[["DFCI_MRN", event, f"tt_{event}"]], on="DFCI_MRN", how="inner")
            .dropna(subset=[event, f"tt_{event}"])
            .copy()
        )
        event_df = event_df.loc[event_df[f"tt_{event}"] > 0].copy()
        if event_df.empty:
            _write_skip_report(skip_fp, scheme, event, "Merged event/risk dataframe has 0 analyzable rows")
            continue

        risk_cols = [f"{modality}_risk_score" for modality in args.modalities]

        for modality in args.modalities:
            modality_col = f"{modality}_risk_score"
            row: dict[str, object] = {"event": event, "modality": modality}
            try:
                metrics, _ = _fit_risk_score_coxph(
                    event_df,
                    [modality_col],
                    event_col=event,
                    tstop_col=f"tt_{event}",
                    max_iter=args.max_iter,
                )
                row.update(metrics)
                row["error"] = ""
            except Exception as exc:
                row.update(
                    {
                        "n": int(len(event_df)),
                        "n_events": int(event_df[event].sum()),
                        "n_censored": int(len(event_df) - event_df[event].sum()),
                        "mean_c_index": np.nan,
                        "mean_auc(t)": np.nan,
                        "mean_ibs": np.nan,
                        "error": str(exc),
                    }
                )
            univariate_rows.append(row)

        joint_row: dict[str, object] = {"event": event}
        try:
            joint_metrics, coef_df = _fit_risk_score_coxph(
                event_df,
                risk_cols,
                event_col=event,
                tstop_col=f"tt_{event}",
                max_iter=args.max_iter,
            )
            joint_row.update(joint_metrics)
            joint_row["error"] = ""

            coef_df["event"] = event
            coef_df["modality"] = coef_df["covariate"].str.removesuffix("_risk_score")
            beta_rows.extend(
                coef_df[["event", "modality", "covariate", "beta"]].to_dict(orient="records")
            )
        except Exception as exc:
            joint_row.update(
                {
                    "n": int(len(event_df)),
                    "n_events": int(event_df[event].sum()),
                    "n_censored": int(len(event_df) - event_df[event].sum()),
                    "mean_c_index": np.nan,
                    "mean_auc(t)": np.nan,
                    "mean_ibs": np.nan,
                    "error": str(exc),
                }
            )
        joint_rows.append(joint_row)

    pd.DataFrame(
        univariate_rows,
        columns=[
            "event",
            "modality",
            "n",
            "n_events",
            "n_censored",
            "mean_c_index",
            "mean_auc(t)",
            "mean_ibs",
            "error",
        ],
    ).to_csv(univar_fp, index=False)
    pd.DataFrame(
        joint_rows,
        columns=[
            "event",
            "n",
            "n_events",
            "n_censored",
            "mean_c_index",
            "mean_auc(t)",
            "mean_ibs",
            "error",
        ],
    ).to_csv(joint_fp, index=False)
    pd.DataFrame(beta_rows, columns=["event", "modality", "covariate", "beta"]).to_csv(beta_fp, index=False)

    with open(meta_fp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme": scheme,
                "modalities": args.modalities,
                "risk_score_dir": risk_root,
                "risk_scores_standardized_before_fit": True,
                "n_requested_events": len(events),
                "n_univariate_rows": len(univariate_rows),
                "n_joint_rows": len(joint_rows),
                "n_joint_beta_rows": len(beta_rows),
            },
            f,
            indent=2,
        )

    print(f"[done] {scheme} -> {univar_fp}")
    print(f"[done] {scheme} -> {joint_fp}")
    print(f"[done] {scheme} -> {beta_fp}")
    print(f"[done] {scheme} -> {meta_fp}")
    if os.path.exists(skip_fp):
        print(f"[done] {scheme} -> {skip_fp}")

    return {
        "scheme": scheme,
        "status": "completed",
        "output_dir": out_dir,
        "n_requested_events": len(events),
        "n_univariate_rows": len(univariate_rows),
        "n_joint_rows": len(joint_rows),
        "n_joint_beta_rows": len(beta_rows),
    }


def main() -> None:
    args = _parse_args()
    args.modalities = _normalize_modalities(args.modalities)
    schemes = sorted(SCHEME_CONFIG) if args.all_schemes else _normalize_modalities(args.scheme)

    results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    multi_scheme = len(schemes) > 1

    for scheme in schemes:
        print(f"[run] scheme={scheme}")
        try:
            results.append(_run_one_scheme(args, scheme, multi_scheme=multi_scheme))
        except Exception as exc:
            print(f"[error] {scheme}: {exc}")
            failures.append((scheme, str(exc)))

    print("[summary]")
    for row in results:
        print(
            f"  {row['scheme']}: {row['status']}"
            + (f", out={row['output_dir']}" if 'output_dir' in row else "")
        )
    for scheme, err in failures:
        print(f"  {scheme}: failed ({err})")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
