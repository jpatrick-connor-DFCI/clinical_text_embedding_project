"""Tests for the Figure 5 gate: does text improve confounding control?

`summarize_ps_diagnostics` decides whether the biomarker arm earns main-figure
status, so the three readings it can return -- success, failure, ambiguous --
are pinned here against constructed diagnostics. The ambiguous case matters
most: a sharper propensity model with a collapsed effective sample is *worse*,
not better, and a summary that reported AUC alone would call it a win.
"""

import os
import sys

import polars as pl
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("zstandard")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.biomarkers.run_IPTW_analysis import _melt_diagnostic  # noqa: E402


def _write_spec(root, cohort, ps_model, cancer_type, *, auc, ess_each,
                smd_struct, smd_embed=None, n_each=500, legacy=False):
    """Write one `{cancer_type}_diagnostics.parquet` for a cohort x ps_model spec.

    `legacy=True` writes the pre-D2/D5 shape: no analyzability section and
    balance keys with no covariate-family prefix.
    """
    d = os.path.join(root, f"IPTW_runs_{cohort}_{ps_model}")
    os.makedirs(d, exist_ok=True)
    frames = [_melt_diagnostic(pl.DataFrame([{
        "N_treated": n_each, "N_control": n_each,
        "events_treated": 200, "events_control": 210,
        "PS_AUC": auc,
        "ESS_ATE_treated": ess_each, "ESS_ATE_control": ess_each,
    }]), "cohort")]

    if legacy:
        bal = pl.DataFrame({"covariate": ["AGE", "GENDER"],
                            "SMD_unweighted": [0.30, 0.28],
                            "SMD_weighted": [smd_struct, smd_struct * 0.5]})
    else:
        bal = pl.DataFrame({
            "covariate": ["structured|AGE", "embedding|emb_1"],
            "SMD_unweighted": [0.30, 0.35],
            "SMD_weighted": [smd_struct, smd_embed],
        })
    frames.append(_melt_diagnostic(bal, "balance_ATE", key_col="covariate"))

    if not legacy:
        ess_frac = (2 * ess_each) / (2 * n_each)
        max_smd = max(smd_struct, smd_embed)
        frames.append(_melt_diagnostic(pl.DataFrame([{
            "analyzable": float(max_smd <= 0.1 and ess_frac >= 0.30),
            "max_smd_ate": max_smd, "n_imbalanced_ate": 0.0,
            "ESS_fraction": ess_frac,
        }]), "analyzability"))

    pl.concat(frames).with_columns(
        pl.lit(cancer_type, dtype=pl.Utf8).alias("cancer_type")
    ).select("cancer_type", "section", "key", "metric", "value").write_parquet(
        os.path.join(d, f"{cancer_type}_diagnostics.parquet"))


@pytest.fixture
def summarizer(tmp_path, monkeypatch):
    """Point the module's BIOMARKER_PATH at a temporary tree."""
    import pipelines.biomarkers.summarize_ps_diagnostics as mod
    root = str(tmp_path / "biomarker_analysis")
    os.makedirs(root, exist_ok=True)
    monkeypatch.setattr(mod, "BIOMARKER_PATH", root)
    monkeypatch.setattr(mod, "OUTPUT_DIR", os.path.join(root, "compiled_results"))
    return mod, root


def _verdict(mod, cohort="cohort1", cancer_type="LUNG"):
    wide = mod.pivot_by_ps_model(mod.collect())
    vdf = mod.interpret(wide)
    row = vdf.filter((pl.col("cohort") == cohort) &
                     (pl.col("cancer_type") == cancer_type))
    return row["verdict"].item()


def test_success_when_auc_rises_and_balance_holds(summarizer):
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_only", "LUNG",
                auc=0.62, ess_each=400, smd_struct=0.09, smd_embed=0.09)
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "LUNG",
                auc=0.71, ess_each=380, smd_struct=0.05, smd_embed=0.06)
    assert _verdict(mod) == "SUCCESS"


def test_failure_when_auc_is_essentially_unchanged(summarizer):
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_only", "LUNG",
                auc=0.640, ess_each=400, smd_struct=0.08, smd_embed=0.08)
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "LUNG",
                auc=0.645, ess_each=390, smd_struct=0.08, smd_embed=0.08)
    assert _verdict(mod) == "FAILURE"


def test_failure_when_auc_rises_but_balance_worsens(summarizer):
    """A sharper PS that balances *worse* is not confounding control."""
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_only", "LUNG",
                auc=0.60, ess_each=400, smd_struct=0.06, smd_embed=0.06)
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "LUNG",
                auc=0.70, ess_each=390, smd_struct=0.09, smd_embed=0.09)
    assert _verdict(mod) == "FAILURE"


def test_ambiguous_when_a_sharper_ps_collapses_the_effective_sample(summarizer):
    """The dangerous case: AUC way up, ESS gone. Better balance bought with
    variance, and in the limit perfect prediction means no overlap at all."""
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_only", "LUNG",
                auc=0.60, ess_each=400, smd_struct=0.09, smd_embed=0.09)
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "LUNG",
                auc=0.88, ess_each=115, smd_struct=0.04, smd_embed=0.04)
    assert _verdict(mod) == "AMBIGUOUS"


def test_incomplete_when_only_one_ps_model_ran(summarizer):
    """No fabricated comparison from a one-sided grid."""
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_only", "LUNG",
                auc=0.62, ess_each=400, smd_struct=0.09, smd_embed=0.09)
    assert _verdict(mod) == "INCOMPLETE"


def test_balance_is_summarized_per_covariate_family(summarizer):
    """D2: a well-balanced structured set must not mask embedding imbalance."""
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "LUNG",
                auc=0.71, ess_each=380, smd_struct=0.02, smd_embed=0.14)
    tidy = mod.collect()
    row = tidy.row(0, named=True)
    assert row["max_smd_weighted_structured"] == pytest.approx(0.02)
    assert row["max_smd_weighted_embedding"] == pytest.approx(0.14)
    assert row["max_smd_weighted_all"] == pytest.approx(0.14)


def test_specs_failing_the_gate_are_reported_not_hidden(summarizer):
    """A spec that writes no results file still appears, flagged.

    Cancer types are discovered from diagnostics rather than results precisely
    so the analyzability failures stay visible.
    """
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "SKIN",
                auc=0.88, ess_each=115, smd_struct=0.04, smd_embed=0.04)
    tidy = mod.collect()
    assert tidy["cancer_type"].to_list() == ["SKIN"]
    assert tidy["analyzable"].item() == 0.0
    assert tidy["results_written"].item() == 0.0


def test_legacy_diagnostics_without_an_analyzability_section_still_read(summarizer):
    mod, root = summarizer
    _write_spec(root, "cohort1", "covariates_only", "LUNG",
                auc=0.61, ess_each=400, smd_struct=0.05, legacy=True)
    _write_spec(root, "cohort1", "covariates_plus_embeddings", "LUNG",
                auc=0.70, ess_each=400, smd_struct=0.04, legacy=True)
    tidy = mod.collect()
    # Unprefixed keys are reported as the structured family, which is what
    # those files contained.
    assert "max_smd_weighted_structured" in tidy.columns
    assert "max_smd_weighted_embedding" not in tidy.columns
    assert _verdict(mod) == "SUCCESS"


def test_empty_tree_returns_an_empty_frame(summarizer):
    mod, _ = summarizer
    assert mod.collect().is_empty()
