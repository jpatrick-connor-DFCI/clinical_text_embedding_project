"""Refresh a manuscript export package from the current four main figures.

The package must already contain figure_index.csv and export_manifest.json.
Supplementary figures retain their original numbering and are restricted to the
selected manuscript set. Removed artwork and captions are archived with the old
index and manifest. Tables and the source manuscript are not modified here.

Example:
    python3 figures/export_desktop_figures.py --package /path/to/export \
        --revision-date 2026-10-12
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "manuscript_drafts" / "manuscript_figures"
FIELDNAMES = [
    "figure", "title", "page", "file", "source_png", "width_px", "height_px", "sha256"
]
SUPPLEMENTARY_FIGURES = (4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20)
MAIN_TITLES = {
    1: "Clinical text embedding, endpoint prediction, and manuscript analysis workflow.",
    2: "Clinical-text discrimination, published-score comparisons, and survival stratification.",
    3: "Predictive performance across clinical and genomic modalities.",
    4: "Exploratory mortality-risk trajectories and subsequent survival.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_write(path: Path, contents: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(contents)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_archive(package: Path, paths: set[Path], revision_date: str) -> tuple[Path, bytes] | None:
    """Prepare an immutable backup before any active package files are changed."""
    if not paths:
        return None
    archive = package / "Archive" / f"supplementary_figures_before_curation_{revision_date}.zip"
    entries = {path.relative_to(package).as_posix(): path.read_bytes()
               for path in paths | {package / "figure_index.csv", package / "export_manifest.json"}}
    if archive.exists():
        with zipfile.ZipFile(archive) as existing:
            saved = {name: existing.read(name) for name in existing.namelist()}
        if saved != entries:
            raise ValueError(f"Existing supplementary archive differs; refusing to overwrite: {archive}")
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, contents in sorted(entries.items()):
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(entry, contents)
    return archive, buffer.getvalue()


def refresh_package(package: Path, source: Path, revision_date: str,
                    include_captions: bool = False) -> dict:
    """Copy current artwork and rebuild package indexes after validating inputs."""
    date.fromisoformat(revision_date)
    package = package.resolve()
    source = source.resolve()
    manifest_path = package / "export_manifest.json"
    index_path = package / "figure_index.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected figure-index columns: {reader.fieldnames}")
        old_rows = list(reader)

    replacements = [
        (
            f"Main Figure {number}",
            title,
            source / "png" / f"figure{number}" / f"figure{number}_cindex.png",
            f"Figures/Main/Figure_{number:02d}.png",
        )
        for number, title in MAIN_TITLES.items()
    ]

    # Validate every source and retained figure before changing the package.
    new_rows = []
    for label, title, source_png, destination in replacements:
        with Image.open(source_png) as artwork:
            if artwork.format != "PNG":
                raise ValueError(f"Expected PNG artwork: {source_png}")
            width, height = artwork.size
            artwork.verify()
        new_rows.append(dict(zip(FIELDNAMES, [
            label, title, "1", destination, source_label(source_png),
            str(width), str(height), sha256(source_png),
        ])))

    retained_rows = []
    removed_paths = set()
    indexed_supplements = set()
    seen_pages = set()
    for row in old_rows:
        if re.fullmatch(r"Main Figure [1-5]", row["figure"]):
            continue
        if not re.fullmatch(r"Supplementary Figure \d+", row["figure"]):
            raise ValueError(f"Unexpected figure label: {row['figure']}")
        number, page = int(row["figure"].split()[-1]), int(row["page"])
        if page < 1 or (number, page) in seen_pages:
            raise ValueError(f"Invalid or duplicate supplementary page: {row['figure']}, page {page}")
        seen_pages.add((number, page))
        path = (package / row["file"]).resolve()
        expected_folder = package / "Figures" / "Supplementary"
        expected_name = rf"Supplementary_Figure_{number:02d}(?:_page_{page})?\.png"
        if path.parent != expected_folder or not re.fullmatch(expected_name, path.name):
            raise ValueError(f"Unexpected supplementary artwork path: {row['file']}")
        indexed_supplements.add(path)
        if number not in SUPPLEMENTARY_FIGURES:
            removed_paths.add(path)
            continue
        with Image.open(path) as artwork:
            if artwork.format != "PNG":
                raise ValueError(f"Expected PNG artwork: {path}")
            row["width_px"], row["height_px"] = map(str, artwork.size)
            artwork.verify()
        row["sha256"] = sha256(path)
        retained_rows.append(row)

    unexpected_supplements = set((package / "Figures" / "Supplementary").glob("*.png")) - indexed_supplements
    if unexpected_supplements:
        raise ValueError(f"Unindexed supplementary artwork: {sorted(path.name for path in unexpected_supplements)}")
    for path in (package / "Figures" / "Captions").glob("Supplementary_Figure_*"):
        match = re.fullmatch(r"Supplementary_Figure_(\d+)(?:_page_\d+)?\.(?:md|txt)", path.name)
        if match and int(match.group(1)) not in SUPPLEMENTARY_FIGURES:
            removed_paths.add(path)

    indexed_main = {f"Figure_{number:02d}.png" for number in range(1, 6)}
    unexpected_main = [
        path.name for path in (package / "Figures" / "Main").glob("*.png")
        if path.name not in indexed_main
    ]
    if unexpected_main:
        raise ValueError(f"Unexpected main-figure files: {unexpected_main}")

    caption_copies = []
    if include_captions:
        for number in MAIN_TITLES:
            stem = f"Figure_{number:02d}"
            available = [source / "captions" / f"figure{number}.{extension}"
                         for extension in ("txt", "md")]
            available = [path for path in available if path.is_file()]
            if not available:
                raise FileNotFoundError(f"No caption found for Figure {number}: {source / 'captions'}")
            caption_copies.extend((path, package / "Figures" / "Captions" / f"{stem}{path.suffix}")
                                  for path in available)

    # Read all inputs and prepare the archive before changing any package file.
    for caption, _ in caption_copies:
        caption.read_bytes()
    archive = prepare_archive(package, removed_paths, revision_date)
    if archive is not None:
        archive_path, archive_bytes = archive
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(archive_path, archive_bytes)

    for _, _, source_png, destination in replacements:
        target = package / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_png != target:
            shutil.copy2(source_png, target)
    (package / "Figures" / "Main" / "Figure_05.png").unlink(missing_ok=True)
    for path in removed_paths:
        path.unlink()
    for caption, target in caption_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(caption, target)

    rows = new_rows + retained_rows
    rows.sort(key=lambda row: (
        row["figure"].startswith("Supplementary"),
        int(row["figure"].split()[-1]),
        int(row["page"]),
    ))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(index_path, output.getvalue().encode("utf-8-sig"))

    supplementary_pages: dict[int, set[int]] = {}
    for row in rows:
        if row["figure"].startswith("Supplementary"):
            number = int(row["figure"].split()[-1])
            supplementary_pages.setdefault(number, set()).add(int(row["page"]))
    manifest.update(
        main_figures=4,
        supplementary_figures=len(supplementary_pages),
        supplementary_figure_ids=sorted(supplementary_pages),
        figure_png_files=len(rows),
        multipage_supplementary_figures=sorted(
            number for number, pages in supplementary_pages.items() if len(pages) > 1
        ),
        main_tables_csv=len(list((package / "Tables" / "Main").glob("*.csv"))),
        supplementary_tables_csv=len(list((package / "Tables" / "Supplementary").glob("*.csv"))),
        figure_revision={
            "date": revision_date,
            "source_root": source_label(source),
            "export_script": "figures/export_desktop_figures.py",
            "render_scripts": [
                "figures/render_workflow.py", "figures/revise_saved_figures.py",
                "figures/render_published_score_panel.py", "R/plot_figure_1.R", "R/plot_figure_2.R",
                "R/plot_figure_3.R", "R/plot_figure_4.R",
            ],
            "description": (
                "Four main figures: full workflow; clinical-text performance with published-score "
                "comparison in panel e; modality comparisons; mortality trajectories. Selected "
                "supplementary figures retain their original IDs: 04, 06, 07, 08, 09, 10, 11, "
                "13, 14, 15, 16, 17, 18, 19, 20. Cohort, cancer-type, stage and data-availability "
                "information is in Supplementary Table 1; endpoint counts are in Supplementary Table 2."
            ),
            "selected_supplementary_figure_ids": list(SUPPLEMENTARY_FIGURES),
            "source_manuscript_updated": False,
            "provenance_note": (
                "source_manuscript, source_sha256, and export_date describe the original manuscript "
                "export. The figure artwork and figure index were revised independently of the "
                "source manuscript. Separate table revisions are recorded in table_revision when present."
            ),
        },
    )
    manifest["files"] = [
        {"file": path.relative_to(package).as_posix(), "bytes": path.stat().st_size,
         "sha256": sha256(path)}
        for path in sorted(package.rglob("*"))
        if path.is_file() and path != manifest_path
        and not any(part.startswith(".") for part in path.relative_to(package).parts)
    ]
    atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, help="Existing exported package")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Rendered manuscript_figures directory")
    parser.add_argument("--revision-date", default=date.today().isoformat(), help="Revision date (YYYY-MM-DD)")
    parser.add_argument("--captions", action="store_true", help="Include available TXT/Markdown captions for revised figures")
    args = parser.parse_args()
    manifest = refresh_package(args.package, args.source, args.revision_date, args.captions)
    print(
        f"Updated {args.package}: {manifest['main_figures']} main figures, "
        f"{manifest['supplementary_figures']} supplementary figures, "
        f"{manifest['figure_png_files']} figure PNG files."
    )


if __name__ == "__main__":
    main()
