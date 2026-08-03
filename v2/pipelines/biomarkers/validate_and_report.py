"""Validate unassessed biomarker hits and update the summary report.

Reads all_findings_with_validation.csv, assigns validation levels to
previously unassessed hits based on published ICI biomarker literature,
and regenerates the Word document with updated summary tables.

Runs locally — does not require cluster access.
"""

import os

from config import DATA_PATH

from validate import load_or_bootstrap_findings_df, validate_findings
from report import load_or_init_report_doc, update_report_document

# ============================================================
# Paths
# ============================================================
DEFAULT_COMPILED_DIR = os.path.join(DATA_PATH, 'biomarker_analysis', 'compiled_results')
LEGACY_COMPILED_DIR = '/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/compiled_results/'


def _resolve_compiled_dir():
    env_dir = os.getenv('COMPILED_DIR')
    if env_dir:
        return env_dir

    preferred = DEFAULT_COMPILED_DIR
    legacy_inputs = [
        os.path.join(LEGACY_COMPILED_DIR, 'all_findings_with_validation.csv.gz'),
        os.path.join(LEGACY_COMPILED_DIR, 'track1_all_significant_hits.csv.gz'),
        os.path.join(LEGACY_COMPILED_DIR, 'track2_all_significant_hits.csv.gz'),
    ]
    preferred_inputs = [
        os.path.join(preferred, 'all_findings_with_validation.csv.gz'),
        os.path.join(preferred, 'track1_all_significant_hits.csv.gz'),
        os.path.join(preferred, 'track2_all_significant_hits.csv.gz'),
    ]

    if not any(os.path.exists(path) for path in preferred_inputs) and any(os.path.exists(path) for path in legacy_inputs):
        return LEGACY_COMPILED_DIR
    return preferred


COMPILED_DIR = _resolve_compiled_dir()
FINDINGS_CSV = os.path.join(COMPILED_DIR, 'all_findings_with_validation.csv.gz')
REPORT_DOCX = os.path.join(COMPILED_DIR, 'ICI_Biomarker_Pipeline_Report.docx')


def main() -> None:
    df = load_or_bootstrap_findings_df(COMPILED_DIR, FINDINGS_CSV)
    df = validate_findings(df, FINDINGS_CSV)

    # ============================================================
    # Update Word document
    # ============================================================
    print(f"\n{'='*60}")
    print("Updating Word document")
    print(f"{'='*60}")

    doc = load_or_init_report_doc(REPORT_DOCX)
    doc = update_report_document(doc, df)

    # Save
    doc.save(REPORT_DOCX)
    print(f"Word document updated: {REPORT_DOCX}")
    print("Done.")


if __name__ == "__main__":
    main()
