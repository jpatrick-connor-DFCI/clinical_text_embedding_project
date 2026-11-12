"""Recompose saved manuscript artwork without refitting models or patient data.

Run after render_workflow.py and render_published_score_panel.py. Original
Figures 1/2 are archived once, so repeated runs do not re-crop revised artwork.
The score panel is plotted from the existing primary-analysis report table.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clinical-figure-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from manuscript_fonts import configure_manuscript_font

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript_drafts/manuscript_figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def read_page(data: bytes):
    return PdfReader(BytesIO(data)).pages[0]


def crop(page, box):
    page = deepcopy(page)
    x0, y0, x1, y1 = box
    page.cropbox.lower_left = (x0, y0)
    page.cropbox.upper_right = (x1, y1)
    result = PageObject.create_blank_page(width=x1 - x0, height=y1 - y0)
    result.merge_transformed_page(page, Transformation().translate(-x0, -y0))
    return result


def write_pdf(page, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(page)
    with path.open("wb") as out:
        writer.write(out)


def overlay(width, height, labels=(), caption=None):
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="none")
    for label, x, y in labels:
        fig.patches.append(plt.Rectangle(
            (x / width, (y - 20) / height), 24 / width, 20 / height,
            transform=fig.transFigure, facecolor="white", edgecolor="none"))
        fig.text((x + 2) / width, (y - 2) / height, label, fontsize=12,
                 fontweight="bold", va="top")
    if caption:
        fig.text(8 / width, (height - 8) / height, caption, fontsize=8,
                 va="top", linespacing=1.25)
    buf = BytesIO()
    fig.savefig(buf, format="pdf", transparent=True)
    plt.close(fig)
    return read_page(buf.getvalue())


def save_art(page, group, stem, caption, renderer, dpi):
    pdf = BASE / "pdf" / group / f"{stem}.pdf"
    write_pdf(page, pdf)
    png = BASE / "png" / group / f"{stem}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(renderer), str(pdf), str(png), str(dpi)], check=True)
    if caption is None:
        return
    for ext in ("md", "txt"):
        dest = BASE / "captions" / f"{group}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(caption + "\n")
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    wrap_width = int(width / 72 * 16)
    wrapped = "\n".join(textwrap.fill(p, width=wrap_width) if p else ""
                        for p in caption.split("\n"))
    caption_height = len(wrapped.splitlines()) * 10.5 + 20
    review = PageObject.create_blank_page(width=width, height=height + caption_height)
    review.merge_transformed_page(page, Transformation().translate(0, caption_height))
    with plt.rc_context():
        if group == "figure1":
            configure_manuscript_font()
        review.merge_page(overlay(width, caption_height, caption=wrapped))
    save_art(review, group, stem + "_captioned", None, renderer, dpi)


def save_figure2(original, score, stage, risk, caption, original_caption, renderer, dpi):
    """Keep original panels a–d together, with published scores below as e."""
    width, height = float(original.mediabox.width), float(original.mediabox.height)
    score_scale = width / float(score.mediabox.width)
    score_height = float(score.mediabox.height) * score_scale
    compiled = PageObject.create_blank_page(width=width, height=height + score_height)
    compiled.merge_transformed_page(score, Transformation().scale(score_scale))
    compiled.merge_transformed_page(original, Transformation().translate(0, score_height))
    compiled.merge_page(overlay(width, height + score_height, [("e", 0, score_height)]))
    extra = "\n\nPanel e uses 1,000 patient-bootstrap resamples. The saved results do not contain an ALBI combined-model estimate."
    # Preserve original data-derived details and endpoint-filter disclosure.
    for paragraph in original_caption.strip().split("\n\n")[2:]:
        extra += "\n\n" + paragraph
    save_art(compiled, "figure2", "figure2_cindex", caption + extra, renderer, dpi)
    for panel, stem in ((stage, "fig2c_cindex"), (risk, "fig2d_cindex"),
                        (score, "fig2e_cindex")):
        save_art(panel, "figure2", stem, None, renderer, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-panel", type=Path, required=True)
    parser.add_argument("--pdf-renderer", type=Path, required=True,
                        help="Executable taking input.pdf output.png dpi; see render_pdf.swift")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    # Fail before changing outputs if prerequisites are missing.
    for path in (args.score_panel, args.pdf_renderer,
                 ROOT / "figures/workflow/clinical_text_workflow.pdf"):
        if not path.is_file():
            parser.error(f"Missing prerequisite: {path}")
    archive = ROOT / "manuscript_drafts/archive/figure_layout_before_2026-10-12.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
            for folder in ("pdf/figure1", "png/figure1", "pdf/figure2", "png/figure2", "captions"):
                for path in (BASE / folder).glob("*"):
                    if path.is_file():
                        out.write(path, path.relative_to(BASE))
    with zipfile.ZipFile(archive) as original:
        old2 = read_page(original.read("pdf/figure2/figure2_cindex.pdf"))
        old_stage = read_page(original.read("pdf/figure2/fig2c_cindex.pdf"))
        old_risk = read_page(original.read("pdf/figure2/fig2d_cindex.pdf"))
        old2_caption = original.read("captions/figure2.txt").decode()

    specs = json.loads((ROOT / "figures/manuscript_captions.json").read_text())
    def caption(number):
        spec = specs[number]
        label = "Supplementary Figure " + number[1:] if number.startswith("S") else "Figure " + number
        return f"{label}. {spec['title']}\n\n{spec['text']}"

    workflow = read_page((ROOT / "figures/workflow/clinical_text_workflow.pdf").read_bytes())
    save_art(workflow, "figure1", "figure1_cindex", caption("1"), args.pdf_renderer, args.dpi)
    # Exact source PNG preserves the workflow's original rendering.
    shutil.copy2(ROOT / "figures/workflow/clinical_text_workflow.png",
                 BASE / "png/figure1/figure1_cindex.png")

    # The pre-schematic archive holds full-sized original cancer/stage panels.
    first_archive = ROOT / "manuscript_drafts/archive/figure1_before_relayout_2026-10-11.zip"
    with zipfile.ZipFile(first_archive) as original:
        prefix = "manuscript_drafts/manuscript_figures/pdf/figure1/"
        original1 = read_page(original.read(prefix + "figure1_cindex.pdf"))
        width, height = float(original1.mediabox.width), float(original1.mediabox.height)
        composition = crop(original1, (0, height / 2, width, height))
        supp_caption = caption("S21") + "\n\nCancer-type cohort: 46,914 patients. Known-stage cohort: 30,827 patients."
        save_art(composition, "figureS21", "figS21_cohort_composition_cindex",
                 supp_caption, args.pdf_renderer, args.dpi)
        for source, stem in (("fig1a_cindex", "figS21a_cancer_types_cindex"),
                             ("fig1b_cindex", "figS21b_stage_counts_cindex")):
            save_art(read_page(original.read(prefix + source + ".pdf")),
                     "figureS21", stem, None, args.pdf_renderer, args.dpi)

    score = read_page(args.score_panel.read_bytes())
    save_figure2(old2, score, old_stage, old_risk, caption("2"), old2_caption,
                 args.pdf_renderer, args.dpi)

    # Superseded main-1 panel files are preserved in the archive and now S21.
    for ext in ("pdf", "png"):
        for path in (BASE / ext / "figure1").glob(f"fig1?_cindex.{ext}"):
            path.unlink()
    print("Updated saved Figures 1/2 and Supplementary Figure 21; Figures 3/4 preserved.")


if __name__ == "__main__":
    main()
