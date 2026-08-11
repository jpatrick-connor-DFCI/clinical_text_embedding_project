"""Loads validation_reference.csv and rebuilds the VALIDATION_REF lookup dict
that used to be ~580 lines of hand-curated literature hardcoded in
validate_and_report.py.

Key shape is preserved exactly as it was in the original dict literal:
  - cancer_type == ""  -> 2-tuple key (gene, mutation_type)
  - cancer_type != ""  -> 3-tuple key (gene, mutation_type, cancer_type)
    (this includes the literal value "pan_cancer" as one possible cancer_type,
    which is a distinct lookup tier from the empty/2-tuple case — see
    validate_hit()'s 3-tier lookup order in pipelines/biomarkers/validate.py).
"""
import csv
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "validation_reference.csv"


def load_validation_reference() -> dict:
    validation_ref: dict = {}
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            gene = row["gene"]
            mutation_type = row["mutation_type"]
            cancer_type = row["cancer_type"]
            key = (gene, mutation_type) if cancer_type == "" else (gene, mutation_type, cancer_type)
            validation_ref[key] = (row["level"], row["notes"])
    return validation_ref
