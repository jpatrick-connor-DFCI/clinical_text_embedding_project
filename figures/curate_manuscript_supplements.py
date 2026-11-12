"""Install the consolidated cohort table and curate an existing manuscript export.

Original table identifiers are preserved. Tables 3 and 4 are incorporated into
Table 1, with their original files and index retained in an immutable archive.
The source Word manuscript is not changed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import io
import json
from pathlib import Path
import zipfile

from export_desktop_figures import (
    DEFAULT_SOURCE, REPO_ROOT, atomic_write, refresh_package, sha256, source_label,
)


TABLE_SOURCE = REPO_ROOT / "manuscript_drafts/manuscript_tables"
TITLE = "Cohort composition and data availability."
NOTES = (
    "Consolidates original Supplementary Tables 1, 3, and 4. Cumulative availability "
    "uses the initial eligible cohort (65,080 patients); cancer types use the text "
    "cohort (46,914); stage uses patients with recognized stage I–IV (30,827). "
    "Counts and displayed percentages are preserved from the source tables; "
    "rounded cancer-type percentages sum to 99.9%. Other pools less frequent "
    "cancer types and differs from the model covariate OTHER. The stage-availability "
    "cohort differs from the 29,002-patient stage-survival cohort, and stage "
    "annotations were not restricted uniformly to the prediction anchor. "
    "Endpoint-specific populations may be smaller. Endpoint counts remain in "
    "Supplementary Table 2. PRS, polygenic risk scores."
)


def csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def curate_tables(package: Path, table_source: Path, revision_date: str) -> None:
    """Preflight, archive, and install the combined table without renumbering."""
    date.fromisoformat(revision_date)
    package, table_source = package.resolve(), table_source.resolve()
    index_path = package / "Tables/table_index.csv"
    manifest_path = package / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or not {"section", "table_number", "filename", "sha256"} <= set(fields):
        raise ValueError("Unexpected table-index schema")
    table1 = [row for row in rows if row["section"] == "Supplementary" and row["table_number"] == "1"]
    if len(table1) != 1:
        raise ValueError("Expected exactly one Supplementary Table 1 index entry")
    replacements = {
        package / f"Tables/Supplementary/Supplementary_Table_01.{ext}":
            (table_source / f"cohort_summary.{ext}").read_bytes()
        for ext in ("csv", "pdf", "docx")
    }
    provenance = (table_source / "cohort_summary_provenance.json").read_bytes()
    json.loads(provenance)
    replacements[package / "Tables/Supplementary/Supplementary_Table_01_provenance.json"] = provenance
    table_data = list(csv.reader(io.StringIO(
        replacements[package / "Tables/Supplementary/Supplementary_Table_01.csv"].decode("utf-8-sig"))))
    if len(table_data) != 24 or not all(len(row) == len(table_data[0]) for row in table_data):
        raise ValueError("Expected a rectangular cohort table with 23 data rows")
    if not replacements[package / "Tables/Supplementary/Supplementary_Table_01.pdf"].startswith(b"%PDF-"):
        raise ValueError("Invalid cohort table PDF")
    with zipfile.ZipFile(io.BytesIO(replacements[package / "Tables/Supplementary/Supplementary_Table_01.docx"])) as document:
        document.read("word/document.xml")

    endpoint_path = package / "Tables/Supplementary/Supplementary_Table_02.csv"
    endpoint_text = endpoint_path.read_text(encoding="utf-8-sig")
    endpoint_text = endpoint_text.replace("Supplementary Figure 1b available endpoints",
                                          "Cohort overview available endpoints")
    replacements[endpoint_path] = endpoint_text.encode("utf-8-sig")
    retired = [package / f"Tables/Supplementary/Supplementary_Table_{n:02d}.csv" for n in (3, 4)]
    archive_path = package / "Archive" / f"cohort_tables_before_consolidation_{revision_date}.zip"
    if not archive_path.exists():
        archive_files = [index_path, manifest_path, endpoint_path,
                         package / "Tables/Supplementary/Supplementary_Table_01.csv", *retired]
        saved = {str(path.relative_to(package)): path.read_bytes() for path in archive_files}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, contents in sorted(saved.items()):
                entry = zipfile.ZipInfo(name)
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, contents)
        archive_bytes = buffer.getvalue()
    else:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Existing cohort-table archive is corrupt")
            for path in retired:
                name = str(path.relative_to(package))
                if path.exists() and path.read_bytes() != archive.read(name):
                    raise ValueError(f"Retired table differs from its archive: {path}")
        archive_bytes = None

    updated = []
    for row in rows:
        if row["section"] == "Supplementary" and row["table_number"] in ("3", "4"):
            continue
        if row is table1[0]:
            row.update(title=TITLE, caption=f"Supplementary Table 1. {TITLE}", notes=NOTES,
                       rows_including_header=str(len(table_data)), columns=str(len(table_data[0])),
                       merged_cells="None; section labels repeat for each CSV row",
                       source_document=source_label(table_source / "cohort_summary.csv"))
        path = package / "Tables" / row["filename"]
        if path not in replacements and not path.is_file():
            raise FileNotFoundError(path)
        updated.append(row)

    # All source data and the previous tables are readable before any writes.
    if archive_bytes is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(archive_path, archive_bytes)
    for path, contents in replacements.items():
        atomic_write(path, contents)
    for path in retired:
        path.unlink(missing_ok=True)
    for row in updated:
        row["sha256"] = sha256(package / "Tables" / row["filename"])
    atomic_write(index_path, csv_bytes(updated, fields))
    manifest["table_revision"] = {
        "date": revision_date,
        "script": "figures/curate_manuscript_supplements.py",
        "consolidated_table": "Supplementary Table 1",
        "source_tables": [1, 3, 4],
        "retired_table_ids": [3, 4],
        "endpoint_counts_table": 2,
        "source_data": source_label(table_source / "cohort_summary.csv"),
        "formats": ["csv", "pdf", "docx"],
        "original_table_numbers_preserved": True,
        "source_manuscript_updated": False,
    }
    manifest["tables_note"] = (
        "Table titles, captions, notes and checksums are in Tables/table_index.csv. "
        "Supplementary Table 1 consolidates cohort/data-availability, cancer-type, "
        "and stage summaries and is provided as CSV, PDF, and editable Word. Original "
        "Tables 1, 3, and 4 are archived; other table numbers are preserved."
    )
    atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--table-source", type=Path, default=TABLE_SOURCE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--revision-date", default=date.today().isoformat())
    args = parser.parse_args()
    curate_tables(args.package, args.table_source, args.revision_date)
    manifest = refresh_package(args.package, args.source, args.revision_date, include_captions=True)
    print(f"Curated {args.package}: {manifest['supplementary_figures']} supplementary figures, "
          f"{manifest['supplementary_tables_csv']} supplementary tables.")


if __name__ == "__main__":
    main()
