"""Render the supplementary cohort table from its preserved CSV or a manuscript.

The source tables are identified by their column headers, not their position.
Counts and displayed percentages are transcribed without recomputing the
underlying analysis. Outputs: structured CSV, one-page vector PDF, editable
Word table, and a JSON provenance record. Requires matplotlib for the PDF;
DOCX creation uses the standard-library OOXML zip writer.

    python3 figures/render_cohort_summary_table.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "manuscript_drafts/manuscript_export/Tables/Supplementary/Supplementary_Table_01.csv"
DEFAULT_OUTPUT = REPO_ROOT / "manuscript_drafts/manuscript_tables"
TITLE = "Supplementary Table 1. Cohort composition and data availability."
GROUPS = [
    ("Cohort and cumulative data availability", "Cumulative inclusion step", 65080, 1),
    ("Cancer-type composition", "Cancer group shown", 46914, 3),
    ("Known major stage", "Known major stage", 30827, 4),
]
NOTES = [
    "Availability counts represent successive restrictions from the 65,080-patient eligible cohort; "
    "each row is a subset of the preceding row. Percentages use that initial denominator. "
    "Endpoint-specific evaluation populations can be smaller.",
    "Cancer-type percentages use the 46,914-patient text cohort. Other pools less frequent cancer "
    "types for display and is not interchangeable with the model covariate category OTHER. "
    "Displayed percentages total 99.9% because of rounding.",
    "Stage percentages use 30,827 patients with recognized stage I–IV. This availability cohort "
    "differs from the 29,002-patient stage–survival cohort. Stage annotations were not restricted "
    "uniformly to the prediction anchor. PRS, polygenic risk scores.",
    "Values preserve the source manuscript's Supplementary Tables 1, 3 and 4 and the saved cohort, "
    "cancer-type and stage artwork. No patient-level analysis was rerun.",
]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
FIELDS = ["Section", "Characteristic", "Patients", "Percentage", "Denominator"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(source: Path) -> tuple[list[dict], list[dict]]:
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Expected consolidated CSV columns {FIELDS}")
            rows = list(reader)
        for row in rows:
            for field in ("Patients", "Denominator"):
                row[field] = int(row[field].replace(",", ""))
        validate_rows(rows)
        return rows, [
            {
                "section": group,
                "denominator": denominator,
                "source_csv_data_rows_one_based": [
                    index for index, row in enumerate(rows, 1) if row["Section"] == group
                ],
            }
            for group, _, denominator, _ in GROUPS
        ]
    with zipfile.ZipFile(source) as document:
        root = ET.fromstring(document.read("word/document.xml"))
    tables = []
    for table in root.findall(".//w:tbl", NS):
        tables.append([
            ["".join(cell.itertext()) for cell in row.findall("w:tc", NS)]
            for row in table.findall("w:tr", NS)
        ])
    rows, provenance = [], []
    for group, header, denominator, original_number in GROUPS:
        matches = [(index, table) for index, table in enumerate(tables) if table[0][0] == header]
        if len(matches) != 1:
            raise ValueError(f"Expected one table with header {header!r}; found {len(matches)}")
        index, table = matches[0]
        for label, count, percentage in table[1:]:
            rows.append({
                "Section": group,
                "Characteristic": label,
                "Patients": int(count.replace(",", "")),
                "Percentage": percentage,
                "Denominator": denominator,
            })
        provenance.append({
            "section": group,
            "source_supplementary_table": original_number,
            "source_docx_table_index_zero_based": index,
            "denominator": denominator,
            "original_rows": table,
        })
    validate_rows(rows)
    return rows, provenance


def validate_rows(rows: list[dict]) -> None:
    if any(row["Section"] not in {group[0] for group in GROUPS} for row in rows):
        raise ValueError("Unexpected cohort-table section")
    for group, _, denominator, original_number in GROUPS:
        entries = [row for row in rows if row["Section"] == group]
        if len(entries) != {1: 8, 3: 11, 4: 4}[original_number]:
            raise ValueError(f"Unexpected number of rows for {group}")
        counts = [row["Patients"] for row in entries]
        if original_number == 1:
            if counts[0] != denominator or counts != sorted(counts, reverse=True):
                raise ValueError("Availability rows are not a cumulative cohort cascade")
        elif sum(counts) != denominator:
            raise ValueError(f"Counts do not sum to the reported denominator for {group}")
        for row in entries:
            if row["Denominator"] != denominator or row["Patients"] < 0:
                raise ValueError(f"Invalid count or denominator: {row}")
            displayed = float(row["Percentage"].rstrip("%"))
            if abs(displayed - 100 * row["Patients"] / denominator) > 0.051:
                raise ValueError(f"Percentage differs from its stated denominator: {row}")


def render_pdf(rows: list[dict], target: Path, preview: Path | None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cohort_table_mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 612)
    ax.set_ylim(0, 792)
    ax.axis("off")
    left, right = 43, 569
    ax.text(left, 750, TITLE, fontsize=11.6, weight="bold", va="top")
    y = 719
    ax.plot((left, right), (y, y), color="#252525", linewidth=0.85)
    ax.text(left + 7, y - 13, "Characteristic", fontsize=9.2, weight="bold", va="center")
    ax.text(477, y - 13, "Patients, n", fontsize=9.2, weight="bold", ha="right", va="center")
    ax.text(right - 7, y - 13, "%", fontsize=9.2, weight="bold", ha="right", va="center")
    y -= 26
    for group, _, denominator, _ in GROUPS:
        ax.add_patch(Rectangle((left, y - 25), right - left, 25, facecolor="#e9edf0", edgecolor="none"))
        ax.text(left + 7, y - 12.5, f"{group} (N = {denominator:,})", fontsize=9.0, weight="bold", va="center")
        y -= 25
        for row in [r for r in rows if r["Section"] == group]:
            ax.text(left + 7, y - 8.6, row["Characteristic"], fontsize=9.1, va="center")
            ax.text(477, y - 8.6, f"{row['Patients']:,}", fontsize=9.1, ha="right", va="center")
            ax.text(right - 7, y - 8.6, row["Percentage"].rstrip("%"), fontsize=9.1, ha="right", va="center")
            y -= 17.2
        ax.plot((left, right), (y, y), color="#b8bdc2", linewidth=0.4)
    y -= 16
    for note in NOTES:
        lines = textwrap.wrap(note, width=118)
        ax.text(left, y, "\n".join(lines), fontsize=7.7, va="top", linespacing=1.3, color="#30343a")
        y -= len(lines) * 10.1 + 5
    if y < 35:
        raise ValueError(f"Table notes overflow the page: y = {y:.1f}")
    fig.savefig(target, metadata={"Title": TITLE, "Creator": "render_cohort_summary_table.py"})
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=150)
    plt.close(fig)


def paragraph(text: str, *, size: int = 19, bold: bool = False, align: str = "left", after: int = 0) -> str:
    return (
        '<w:p><w:pPr><w:spacing w:before="0" w:after="' + str(after)
        + '"/><w:jc w:val="' + align + '"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" '
        + 'w:hAnsi="Arial"/><w:sz w:val="' + str(size) + '"/>'
        + ('<w:b/>' if bold else '') + '</w:rPr><w:t xml:space="preserve">'
        + escape(text) + '</w:t></w:r></w:p>'
    )


def word_row(texts: list[str], *, heading: bool = False, section: bool = False) -> str:
    widths = [7480, 1800, 1240]
    cells = []
    for index, value in enumerate(texts):
        props = '<w:tcW w:w="' + str(sum(widths) if section else widths[index]) + '" w:type="dxa"/>'
        if section:
            props += '<w:gridSpan w:val="3"/><w:shd w:fill="E9EDF0"/>'
        props += '<w:tcMar><w:top w:w="40" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
        props += '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
        cells.append('<w:tc><w:tcPr>' + props + '</w:tcPr>' + paragraph(
            value, bold=heading or section, align="left" if index == 0 else "right"
        ) + '</w:tc>')
    tr_props = '<w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if heading else '') + '</w:trPr>'
    return '<w:tr>' + tr_props + ''.join(cells) + '</w:tr>'


def render_docx(rows: list[dict], target: Path) -> None:
    table = '<w:tbl><w:tblPr><w:tblW w:w="10520" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
    table += '<w:tblBorders><w:top w:val="single" w:sz="8" w:color="252525"/>'
    table += '<w:bottom w:val="single" w:sz="8" w:color="252525"/></w:tblBorders></w:tblPr>'
    table += '<w:tblGrid><w:gridCol w:w="7480"/><w:gridCol w:w="1800"/><w:gridCol w:w="1240"/></w:tblGrid>'
    table += word_row(["Characteristic", "Patients, n", "%"], heading=True)
    for group, _, denominator, _ in GROUPS:
        table += word_row([f"{group} (N = {denominator:,})"], section=True)
        for row in [r for r in rows if r["Section"] == group]:
            table += word_row([row["Characteristic"], f"{row['Patients']:,}", row["Percentage"].rstrip("%")])
    table += '</w:tbl>'
    body = paragraph(TITLE, size=24, bold=True, after=160) + table + paragraph("", size=8)
    body += ''.join(paragraph(note, size=16, after=80) for note in NOTES)
    body += '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="840" w:right="860" '
    body += 'w:bottom="840" w:left="860" w:header="400" w:footer="400" w:gutter="0"/></w:sectPr>'
    document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + '<w:document xmlns:w="'
    document += NS["w"] + '"><w:body>' + body + '</w:body></w:document>'
    content_types = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    content_types += '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    content_types += '<Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" '
    content_types += 'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    relationships = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    relationships += '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    relationships += 'Target="word/document.xml"/></Relationships>'
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as document_zip:
        document_zip.writestr("[Content_Types].xml", content_types)
        document_zip.writestr("_rels/.rels", relationships)
        document_zip.writestr("word/document.xml", document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, sections = source_rows(source)
    csv_path = output / "cohort_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    pdf_path, docx_path = output / "cohort_summary.pdf", output / "cohort_summary.docx"
    render_pdf(rows, pdf_path, args.preview)
    render_docx(rows, docx_path)
    sources = [
        REPO_ROOT / "manuscript_drafts/manuscript_figures/pdf/figure0/fig0a.pdf",
        REPO_ROOT / "manuscript_drafts/manuscript_figures/pdf/figureS1/figS1_cohort_endpoints_cindex.pdf",
        REPO_ROOT / "manuscript_drafts/manuscript_figures/pdf/figureS21/figS21_cohort_composition_cindex.pdf",
    ]
    source_kind = "source_csv" if source.suffix.lower() == ".csv" else "source_document"
    provenance = {
        "title": TITLE,
        source_kind: str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        f"{source_kind}_sha256": sha256(source),
        "sections": sections,
        "notes": NOTES,
        "numeric_policy": "Preserve source counts and displayed percentages; no patient-level reanalysis.",
        "cross_checked_artwork": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)}
            for path in sources if path.exists()
        ],
        "validation": {
            "data_rows": len(rows),
            "cancer_type_count_sum": sum(r["Patients"] for r in rows if r["Section"] == GROUPS[1][0]),
            "stage_count_sum": sum(r["Patients"] for r in rows if r["Section"] == GROUPS[2][0]),
            "cancer_type_displayed_percentage_sum": round(sum(float(r["Percentage"].rstrip("%")) for r in rows if r["Section"] == GROUPS[1][0]), 1),
            "endpoint_counts": "Retained separately in Supplementary Table 2; not patient counts.",
        },
        "output_sha256": {path.name: sha256(path) for path in (csv_path, pdf_path, docx_path)},
    }
    if source_kind == "source_csv":
        upstream = source.with_name(source.stem + "_provenance.json")
        if upstream.exists():
            provenance["upstream_provenance"] = {
                "path": str(upstream.relative_to(REPO_ROOT)) if upstream.is_relative_to(REPO_ROOT) else str(upstream),
                "sha256": sha256(upstream),
            }
    (output / "cohort_summary_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(rows)} data rows to {output}")


if __name__ == "__main__":
    main()
