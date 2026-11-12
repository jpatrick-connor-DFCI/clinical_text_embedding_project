"""Verify supplemental curation, recoverable removal, and export provenance."""

import csv
import hashlib
import json
import zipfile

from PIL import Image, UnidentifiedImageError
import pytest

from figures.export_desktop_figures import FIELDNAMES, SUPPLEMENTARY_FIGURES, refresh_package


def make_package(tmp_path):
    package = tmp_path / "package"
    source = tmp_path / "source"
    rows = []
    for number in range(1, 6):
        path = package / "Figures" / "Main" / f"Figure_{number:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), "white").save(path)
        rows.append(dict(zip(FIELDNAMES, [
            f"Main Figure {number}", "Old title", "1", str(path.relative_to(package)),
            "old/source.png", "12", "10", "old-hash",
        ])))
    for number, pages in ((4, (1,)), (5, (1, 2)), (15, (1, 2)), (21, (1,))):
        for page in pages:
            suffix = f"_page_{page}" if len(pages) > 1 else ""
            path = package / "Figures" / "Supplementary" / f"Supplementary_Figure_{number:02d}{suffix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (15 + number, 9 + page), "blue").save(path)
            rows.append(dict(zip(FIELDNAMES, [
                f"Supplementary Figure {number}", "Existing supplementary figure", str(page),
                str(path.relative_to(package)), "existing/source.png", "15", "9", "old-hash",
            ])))
    for number in (4, 21):
        path = package / "Figures" / "Captions" / f"Supplementary_Figure_{number:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Original caption for supplementary figure {number}.\n")
    for relative in ("Tables/Main/Table_01.csv", "Tables/Supplementary/Supplementary_Table_03.csv"):
        table = package / relative
        table.parent.mkdir(parents=True)
        table.write_text("label,value\noriginal,42\n")
    with (package / "figure_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    (package / "export_manifest.json").write_text(json.dumps({
        "source_manuscript": "original.docx", "source_sha256": "original-document-hash",
        "export_date": "2026-10-01", "files": [],
        "table_revision": {"description": "Independent table revision"},
    }))
    for number in range(1, 5):
        path = source / "png" / f"figure{number}" / f"figure{number}_cindex.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (30 + number, 20), "red").save(path)
    # No S21 source exists: it must never be required or added back to the package.
    return package, source


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_refresh_curates_supplements_archives_removals_and_preserves_provenance(tmp_path):
    package, source = make_package(tmp_path)
    original = snapshot(package)
    manifest = refresh_package(package, source, "2026-10-12")

    assert manifest["source_manuscript"] == "original.docx"
    assert manifest["source_sha256"] == "original-document-hash"
    assert manifest["export_date"] == "2026-10-01"
    assert manifest["figure_revision"]["source_manuscript_updated"] is False
    assert manifest["figure_revision"]["selected_supplementary_figure_ids"] == list(SUPPLEMENTARY_FIGURES)
    assert manifest["table_revision"] == {"description": "Independent table revision"}
    assert manifest["main_figures"] == 4
    assert manifest["supplementary_figures"] == 2
    assert manifest["supplementary_figure_ids"] == [4, 15]
    assert manifest["figure_png_files"] == 7
    assert manifest["multipage_supplementary_figures"] == [15]
    assert manifest["main_tables_csv"] == 1
    assert manifest["supplementary_tables_csv"] == 1
    assert not (package / "Figures/Main/Figure_05.png").exists()

    excluded = {
        "Figures/Supplementary/Supplementary_Figure_05_page_1.png",
        "Figures/Supplementary/Supplementary_Figure_05_page_2.png",
        "Figures/Supplementary/Supplementary_Figure_21.png",
        "Figures/Captions/Supplementary_Figure_21.txt",
    }
    archive_path = package / "Archive/supplementary_figures_before_curation_2026-10-12.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == excluded | {"figure_index.csv", "export_manifest.json"}
        for name in archive.namelist():
            assert archive.read(name) == original[name]
    for name in excluded:
        assert not (package / name).exists()
    for path, data in original.items():
        if path not in excluded and path.startswith(("Tables/", "Figures/Supplementary/", "Figures/Captions/")):
            assert (package / path).read_bytes() == data
    for entry in manifest["files"]:
        data = (package / entry["file"]).read_bytes()
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    with (package / "figure_index.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert rows[0]["width_px"] == "31"
    assert [(row["figure"], row["page"]) for row in rows[4:]] == [
        ("Supplementary Figure 4", "1"),
        ("Supplementary Figure 15", "1"),
        ("Supplementary Figure 15", "2"),
    ]
    first_revision = snapshot(package)
    refresh_package(package, source, "2026-10-12")
    assert snapshot(package) == first_revision


@pytest.mark.parametrize("missing", ["main", "retained", "excluded"])
def test_missing_artwork_leaves_package_untouched(tmp_path, missing):
    package, source = make_package(tmp_path)
    paths = {
        "main": source / "png/figure4/figure4_cindex.png",
        "retained": package / "Figures/Supplementary/Supplementary_Figure_15_page_2.png",
        "excluded": package / "Figures/Supplementary/Supplementary_Figure_21.png",
    }
    paths[missing].unlink()
    original = snapshot(package)
    with pytest.raises(FileNotFoundError):
        refresh_package(package, source, "2026-10-12")
    assert snapshot(package) == original


def test_corrupt_retained_artwork_leaves_package_untouched(tmp_path):
    package, source = make_package(tmp_path)
    (package / "Figures/Supplementary/Supplementary_Figure_04.png").write_bytes(b"not a PNG")
    original = snapshot(package)
    with pytest.raises(UnidentifiedImageError):
        refresh_package(package, source, "2026-10-12")
    assert snapshot(package) == original


def test_conflicting_archive_is_not_overwritten(tmp_path):
    package, source = make_package(tmp_path)
    archive = package / "Archive/supplementary_figures_before_curation_2026-10-12.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("figure_index.csv", "Earlier, different archive")
    original = snapshot(package)
    with pytest.raises(ValueError, match="archive differs"):
        refresh_package(package, source, "2026-10-12")
    assert snapshot(package) == original


def test_optional_main_captions_are_preflighted_and_manifested(tmp_path):
    package, source = make_package(tmp_path)
    original = snapshot(package)
    with pytest.raises(FileNotFoundError, match="No caption found"):
        refresh_package(package, source, "2026-10-12", include_captions=True)
    assert snapshot(package) == original

    (source / "captions").mkdir()
    for number in (1, 2, 3, 4):
        (source / "captions" / f"figure{number}.txt").write_text(f"Figure {number} caption.\n")
    manifest = refresh_package(package, source, "2026-10-12", include_captions=True)
    assert not (package / "Figures/Captions/Supplementary_Figure_21.txt").exists()
    assert (package / "Figures/Captions/Figure_04.txt").read_text() == "Figure 4 caption.\n"
    assert len([entry for entry in manifest["files"] if entry["file"].startswith("Figures/Captions/")]) == 5
