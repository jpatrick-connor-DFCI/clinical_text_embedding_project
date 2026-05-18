"""Build Slurm Manifests script for model training workflows."""

import argparse
import os
from pathlib import Path

from slurm_array_utils import (
    MIN_EVENTS_FOR_CV,
    MIN_NON_EVENTS_FOR_CV,
    SCHEME_CONFIG,
    filter_event_rows,
    get_events_from_df,
    get_output_dir,
    load_embedding_prediction_df,
)

MODALITIES = ["stage", "treatment", "labs", "somatic", "prs", "text"]
FULL_COHORT_FILES = ["text_test.csv", "text_val.csv", "base_test.csv", "base_val.csv"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "bash_scripts" / "slurm_manifests"


def _event_has_enough_cases(pred_df, event: str, min_events: int = MIN_EVENTS_FOR_CV) -> bool:
    """Check if an event has enough positive and censored cases for CV."""
    event_df = filter_event_rows(pred_df, event)
    if event_df.empty:
        return False
    n_events = int(event_df[event].sum())
    n_non_events = len(event_df) - n_events
    return n_events >= min_events and n_non_events >= MIN_NON_EVENTS_FOR_CV


def _full_cohort_complete(scheme: str, event: str) -> bool:
    """Check if all 4 full-cohort result CSVs exist for a scheme/event."""
    out_dir = os.path.join(get_output_dir(scheme, "full_cohort"), event)
    return all(os.path.exists(os.path.join(out_dir, f)) for f in FULL_COHORT_FILES)


def _feature_comp_complete(scheme: str, event: str) -> bool:
    """Check if all modality result CSVs and held-out risk scores exist for a scheme/event."""
    out_dir = os.path.join(get_output_dir(scheme, "feature_comps"), event)
    risk_dir = os.path.normpath(os.path.join(
        get_output_dir(scheme, "feature_comps"), "..", "held_out_risk_scores", event
    ))
    grid_done = all(
        os.path.exists(os.path.join(out_dir, f"{mod}_{split}.csv"))
        for mod in MODALITIES
        for split in ("test", "val")
    )
    risk_done = all(
        os.path.exists(os.path.join(risk_dir, f"{mod}_risk_scores.csv"))
        for mod in MODALITIES
    )
    return grid_done and risk_done


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TSV manifests for Slurm array event-level training.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where manifest .tsv files are written.",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=sorted(SCHEME_CONFIG.keys()),
        choices=sorted(SCHEME_CONFIG.keys()),
        help="Schemes to include.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Exclude tasks whose result CSVs already exist in the data path.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help=(
            "Minimum number of positive events required to include a task. "
            f"Defaults to MIN_EVENTS_FOR_CV ({MIN_EVENTS_FOR_CV}) from slurm_array_utils."
        ),
    )
    return parser.parse_args()


def _write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    full_rows: list[str] = []
    full_skipped = 0
    feature_rows: list[str] = []
    feature_skipped = 0
    too_few_events = 0

    for scheme in args.schemes:
        pred_df = load_embedding_prediction_df(scheme)
        events = get_events_from_df(pred_df)
        for event in events:
            if args.min_events is not None and not _event_has_enough_cases(pred_df, event, args.min_events):
                too_few_events += 1
                continue
            if args.skip_completed and _full_cohort_complete(scheme, event):
                full_skipped += 1
            else:
                full_rows.append(f"{scheme}\t{event}")
            if args.skip_completed and _feature_comp_complete(scheme, event):
                feature_skipped += 1
            else:
                feature_rows.append(f"{scheme}\t{event}")

    full_fp = os.path.join(args.output_dir, "full_cohort_tasks.tsv")
    feature_fp = os.path.join(args.output_dir, "feature_comp_tasks.tsv")

    _write_lines(full_fp, full_rows)
    _write_lines(feature_fp, feature_rows)

    print(f"[wrote] {full_fp} ({len(full_rows)} rows)")
    print(f"[wrote] {feature_fp} ({len(feature_rows)} rows)")
    if args.min_events is not None:
        print(f"[min-events] {too_few_events} events excluded (fewer than {args.min_events} positive cases)")
    if args.skip_completed:
        print(f"[skip-completed] full_cohort: {full_skipped} already done, {len(full_rows)} remaining")
        print(f"[skip-completed] feature_comp: {feature_skipped} already done, {len(feature_rows)} remaining")


if __name__ == "__main__":
    main()
