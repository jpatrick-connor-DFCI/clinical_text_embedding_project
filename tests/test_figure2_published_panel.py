"""Regression checks for the local Figure 2e aggregate-results renderer."""

from figures.render_published_score_panel import (
    MODELS, SCORES, render_panel, score_rows, select_estimates, unavailable_estimates,
)
import csv

import pytest


def estimate(score, source="calculated", model="published", n=100, **overrides):
    row = dict(score=score, source=source, model=model, n_patients=str(n), n_events="50",
               cindex="0.72", ci_lower="0.68", ci_upper="0.76")
    row.update(overrides)
    return row


def test_requested_scores_primary_cohorts_and_sources():
    records = [estimate(score, model=model) for score in reversed(SCORES) for model in MODELS]
    records += [estimate("ipi", source="regex"), estimate("imdc", source="regex", n=34),
                estimate("imdc_noecog"), estimate("mskcc_noecog"), estimate("ipi_noecog"),
                estimate("capra_mod"), estimate("mgps", anchor="sequencing"),
                estimate("mgps", lab_window_days=90), estimate("mgps", stratum="LUNG"),
                estimate("mgps", status="failed"), estimate("mgps", model="other")]
    selected = select_estimates(records)
    assert list(dict.fromkeys(row["score"] for row in selected)) == list(SCORES)
    assert len(selected) == 31
    assert ("ipi", "regex") in score_rows(selected)
    assert ("imdc", "regex") not in score_rows(selected)
    assert all(row["cindex"] == .72 and row["ci_lower"] == .68 and row["ci_upper"] == .76
               for row in selected)
    assert SCORES["imdc"] == "IMDC (with ECOG)"
    assert SCORES["mskcc"] == "MSKCC (with ECOG)"


def test_missing_combined_is_disclosed_and_duplicates_are_rejected():
    rows = [estimate("albi"), estimate("albi", model="text")]
    selected = select_estimates(rows)
    assert unavailable_estimates(selected) == [("albi", "calculated", "published+text")]
    assert len(selected) == 2
    with pytest.raises(ValueError, match="Duplicate score/source/model"):
        select_estimates(rows + rows)


def test_render_preserves_pdf_labels_and_discloses_missing_estimate(tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    records = [estimate(score, model=model) for score in SCORES for model in MODELS
               if not (score == "albi" and model == "published+text")]
    table = tmp_path / "report.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    outputs = render_panel(table, tmp_path / "fig2e", dpi=72)
    assert all(path.stat().st_size > 0 for path in outputs)
    page = pypdf.PdfReader(outputs[0]).pages[0]
    assert float(page.mediabox.width) == pytest.approx(720)
    assert float(page.mediabox.height) == pytest.approx(439.2)
    text = page.extract_text()
    for label in SCORES.values():
        assert label in text
    assert "Unavailable estimates: ALBI (calculated), Published + text." in text
    assert "ECOG-free" not in text
