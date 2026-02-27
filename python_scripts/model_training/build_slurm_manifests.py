"""Build Slurm Manifests script for model training workflows."""

import argparse
import os
from pathlib import Path

from slurm_array_utils import SCHEME_CONFIG, get_events_from_df, load_embedding_prediction_df

MODALITIES = ["stage", "treatment", "labs", "somatic", "prs", "text"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "bash_scripts" / "slurm_manifests"


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
    return parser.parse_args()


def _write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    full_rows: list[str] = []
    feature_rows: list[str] = []

    for scheme in args.schemes:
        pred_df = load_embedding_prediction_df(scheme)
        events = get_events_from_df(pred_df)
        for event in events:
            full_rows.append(f"{scheme}\t{event}")
            for modality in MODALITIES:
                feature_rows.append(f"{scheme}\t{event}\t{modality}")

    full_fp = os.path.join(args.output_dir, "full_cohort_tasks.tsv")
    feature_fp = os.path.join(args.output_dir, "feature_comp_tasks.tsv")

    _write_lines(full_fp, full_rows)
    _write_lines(feature_fp, feature_rows)

    print(f"[wrote] {full_fp} ({len(full_rows)} rows)")
    print(f"[wrote] {feature_fp} ({len(feature_rows)} rows)")


if __name__ == "__main__":
    main()
