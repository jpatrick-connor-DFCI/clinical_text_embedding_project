"""Check paired, fold-aware survival comparisons and their source-data contracts."""

import gzip
import sys

import numpy as np
import polars as pl
import pytest

from figures.prep import within_cancer as prep


def _paired(times, events, text, other, *, text_folds=None, other_folds=None):
    n = len(times)
    return pl.DataFrame({
        "DFCI_MRN": [str(i) for i in range(n)],
        "time": times,
        "event_flag": events,
        "text_score": text,
        "comparator_score": other,
        "text_fold": text_folds if text_folds is not None else [0] * n,
        "comparator_fold": other_folds if other_folds is not None else [0] * n,
        "cancer_type": ["Reference cancer"] * n,
    })


def _evaluate(frame, **kwargs):
    return prep.evaluate_comparison(
        frame, scheme="death_met", event="death", comparator="base",
        min_patients=kwargs.get("min_patients", 2),
        min_events=kwargs.get("min_events", 1),
    )


def test_comparable_pair_weighting_and_fold_offsets():
    # Three concordant pairs in fold 0 and six discordant pairs in fold 1.
    frame = _paired(
        [1., 2., 3., 1., 2., 3., 4.], [1.] * 7,
        [3., 2., 1., 1., 2., 3., 4.], [1., 2., 3., 4., 3., 2., 1.],
        text_folds=[0, 0, 0, 1, 1, 1, 1],
        other_folds=[1, 1, 1, 0, 0, 0, 0],
    )
    result = _evaluate(frame).row(0, named=True)
    assert result["text_cindex"] == pytest.approx(1 / 3)
    assert result["comparator_cindex"] == pytest.approx(2 / 3)
    assert result["delta_cindex"] == pytest.approx(-1 / 3)
    assert result["n_comparable_pairs"] == 9
    assert result["n_fold_blocks"] == 2
    # Cox score locations can differ by fitted model. No cross-fold comparison
    # should be introduced by combining those scores in the preparation step.
    shifted = frame.with_columns(
        (pl.col("text_score") + 1000 * pl.col("text_fold")).alias("text_score"),
        (pl.col("comparator_score") - 1000 * pl.col("comparator_fold")).alias("comparator_score"),
    )
    assert _evaluate(shifted).equals(_evaluate(frame))


def test_both_models_fold_partitions_define_comparable_pairs():
    frame = _paired(
        [1., 2., 3., 4.], [1.] * 4, [4., 3., 2., 1.], [1., 2., 3., 4.],
        text_folds=[0, 0, 1, 1], other_folds=[0, 1, 0, 1],
    )
    # Each pair of model fold IDs occurs once; comparing any two patients would
    # compare predictions from different fits for at least one of the models.
    result = _evaluate(frame).row(0, named=True)
    assert result["status"] == "no_comparable_pairs"
    assert result["n_comparable_pairs"] == result["n_fold_blocks"] == 0
    assert result["text_cindex"] is None


def test_patient_pairs_never_cross_cancer_types():
    frame = _paired([1., 3., 2., 4.], [1.] * 4, [4., 3., 1., 2.], [1., 2., 4., 3.])
    frame = frame.with_columns(pl.Series("cancer_type", ["Breast", "Breast", "Lung", "Lung"]))
    result = _evaluate(frame)
    assert result["cancer_type"].to_list() == ["Breast", "Lung"]
    assert result["n_comparable_pairs"].to_list() == [1, 1]
    assert result["delta_cindex"].to_list() == [1., -1.]


@pytest.mark.parametrize(
    "times,events,text,other,expected_text,expected_other,expected_pairs",
    [
        ([1., 2., 3., 4.], [1., 0., 1., 0.], [3., 2., 2., 1.], [1., 1., 2., 3.], 4., .5, 4),
        # Same-time event/censor pairs count; tied scores contribute one half.
        ([1., 1., 2.], [1., 0., 1.], [3., 3., 1.], [1., 2., 3.], 1.5, 0., 2),
        # Earlier censoring supplies no comparable pair with later events.
        ([1., 2.], [0., 1.], [2., 1.], [1., 2.], 0., 0., 0),
        ([1., 1.], [1., 1.], [2., 1.], [1., 2.], 0., 0., 0),
    ],
)
def test_hand_calculated_censoring_and_ties(
    times, events, text, other, expected_text, expected_other, expected_pairs,
):
    assert prep._block_pair_counts(_paired(times, events, text, other)) == (
        expected_text, expected_other, expected_pairs
    )


@pytest.mark.parametrize(
    "min_patients,min_events,status",
    [(5, 1, "too_few_patients"), (2, 3, "too_few_events"), (2, 1, "ok")],
)
def test_sparse_strata_are_retained_with_thresholds(min_patients, min_events, status):
    frame = _paired([1., 2., 3., 4.], [1., 0., 1., 0.], [4., 3., 2., 1.], [1., 2., 3., 4.])
    row = _evaluate(frame, min_patients=min_patients, min_events=min_events).row(0, named=True)
    assert row["status"] == status
    assert row["min_patients_required"] == min_patients
    assert row["min_events_required"] == min_events
    assert row["n_patients"] == 4
    assert row["n_events"] == 2
    assert (row["delta_cindex"] is None) == (status != "ok")


def test_matching_uses_same_patients_and_audits_invalid_rows():
    ids = [str(i) for i in range(10)]
    text = pl.DataFrame({
        "DFCI_MRN": ids + ["text_only"],
        "text_score": [10., 9., 8., 7., 6., np.inf, 4., 3., 2., 1., 0.],
        "text_fold": [0.] * 11,
    })
    other = pl.DataFrame({
        "DFCI_MRN": ids + ["other_only"],
        "comparator_score": [float(i) for i in range(11)],
        "comparator_fold": [0., 0., 0., 0., 0., 0., .5, 0., 0., 0., 0.],
    })
    outcomes = pl.DataFrame({
        "DFCI_MRN": ids[:-1],
        "time": [1., 2., 0., np.nan, 5., 6., 7., 8., 9.],
        "event_flag": [1., 1., 1., 1., 2., 1., 1., 1., 1.],
    })
    cancer = pl.DataFrame({
        "DFCI_MRN": ids,
        "cancer_type": ["Reference cancer"] * 7 + [None, "", "Reference cancer"],
    })
    matched, dropped = prep._matched_patients(text, other, outcomes, cancer)
    assert matched["DFCI_MRN"].to_list() == ["0", "1"]
    assert dropped == {
        "unmatched_text_patients": 1,
        "unmatched_comparator_patients": 1,
        "missing_outcomes": 1,
        "invalid_outcomes": 3,
        "invalid_scores": 1,
        "invalid_fold_ids": 1,
        "missing_cancer_type": 2,
    }
    row = _evaluate(matched).row(0, named=True)
    assert (row["text_cindex"], row["comparator_cindex"], row["n_patients"]) == (1., 0., 2)


@pytest.mark.parametrize("ids", [["1", "1"], ["1", None], ["1", " "]])
def test_ambiguous_patient_ids_rejected(ids):
    with pytest.raises(ValueError, match="patient IDs"):
        prep._validate_ids(pl.DataFrame({"DFCI_MRN": ids}), "fixture")


@pytest.fixture
def score_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "SCHEMES", ["death_met"])
    monkeypatch.setattr(prep, "SURV_PATH", str(tmp_path))
    monkeypatch.setattr(prep, "FEATURE_PATH", str(tmp_path))
    monkeypatch.setattr(prep, "scheme_results_dir", lambda scheme: str(tmp_path / scheme))
    ids = [str(i) for i in range(6)]
    labels = pl.DataFrame({
        "DFCI_MRN": ids,
        "CANCER_TYPE": ["Reference cancer"] * 6,
        # Reference cancer has no positive dummy; argmax would incorrectly label it Lung.
        "CANCER_TYPE_LUNG": [0] * 6,
    })
    with gzip.open(tmp_path / "cancer_type_df.csv.gz", "wt") as handle:
        handle.write(labels.write_csv())
    pl.DataFrame({
        "DFCI_MRN": ids, "death": [1] * 6, "tt_death": [1., 2., 3., 4., 5., 6.],
        "EMBEDDING_0": [np.nan] * 6,
    }).write_parquet(tmp_path / prep.embedding_file("death_met"))
    full = tmp_path / "death_met" / "full_cohort_risk_scores" / "death"
    features = tmp_path / "death_met" / "held_out_risk_scores" / "death"
    full.mkdir(parents=True)
    features.mkdir(parents=True)

    def write(directory, modality, scores, patient_ids=ids, folds=None):
        data = {"DFCI_MRN": patient_ids, f"{modality}_risk_score": scores}
        data["outer_fold"] = folds if folds is not None else [0] * len(patient_ids)
        pl.DataFrame(data).write_csv(directory / f"{modality}_risk_scores.csv")

    write(full, "text", [6., 5., 4., 3., 2., 1.])
    write(full, "base", [1., 2., 3., 4., 5., 6.])
    write(features, "text", [1., 2., 3., 4., 5., 6.])
    write(features, "stage", [6., 5., 4., 3., 2.], patient_ids=ids[:5])
    return tmp_path, full, features


def test_distinct_sources_raw_cancer_mapping_and_missing_modalities(score_tree):
    fig2, fig3, audit = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert fig2.schema == fig3.schema == pl.Schema(prep.RESULT_SCHEMA)
    assert fig2["cancer_type"].to_list() == fig3["cancer_type"].to_list() == ["Reference cancer"]
    assert fig2["delta_cindex"].to_list() == [1.]
    assert fig3["delta_cindex"].to_list() == [-1.]
    assert fig2["n_patients"].to_list() == [6]
    assert fig3["n_patients"].to_list() == [5]
    assert fig3["comparator"].to_list() == ["stage"]
    missing = audit.filter(pl.col("status") == "invalid_comparator_input")
    assert set(missing["comparator"]) == set(prep.MODALITY_ORDER) - {"text", "stage"}
    assert audit.filter(pl.col("status") == "unmatched_text_patients")["n_rows"].to_list() == [1]


def test_legacy_scores_missing_fold_are_audited_without_pooled_fallback(score_tree, caplog):
    _, full, _ = score_tree
    path = full / "text_risk_scores.csv"
    pl.read_csv(path).drop("outer_fold").write_csv(path)
    fig2, fig3, audit = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert fig2.is_empty()
    assert not fig3.is_empty()
    assert "missing outer_fold" in caplog.text
    assert audit.filter(pl.col("status") == "invalid_text_or_outcome_input").height == 1


def test_duplicate_predictions_skip_comparator_and_record_reason(score_tree):
    _, _, features = score_tree
    path = features / "stage_risk_scores.csv"
    scores = pl.read_csv(path)
    pl.concat([scores, scores.head(1)]).write_csv(path)
    fig2, fig3, audit = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert not fig2.is_empty()
    assert fig3.is_empty()
    assert audit.filter(pl.col("detail").str.contains("duplicate patient IDs")).height == 1


def test_missing_inputs_preserve_typed_schemas_and_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "FEATURE_PATH", str(tmp_path))
    fig2, fig3, audit = prep.prepare_within_cancer()
    assert fig2.is_empty() and fig3.is_empty()
    assert fig2.schema == fig3.schema == pl.Schema(prep.RESULT_SCHEMA)
    assert audit.schema == pl.Schema(prep.AUDIT_SCHEMA)
    assert audit["status"].to_list() == ["invalid_cancer_input"]


def test_missing_raw_cancer_label_cannot_fall_back_to_dummies(score_tree):
    root, _, _ = score_tree
    path = root / "cancer_type_df.csv.gz"
    labels = pl.read_csv(path).drop("CANCER_TYPE")
    with gzip.open(path, "wt") as handle:
        handle.write(labels.write_csv())
    fig2, fig3, audit = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert fig2.is_empty() and fig3.is_empty()
    assert audit["status"].to_list() == ["invalid_cancer_input"]
    assert "CANCER_TYPE" in audit["detail"][0]


def test_generic_score_column_supported_but_fold_still_required(tmp_path):
    path = tmp_path / "scores.csv"
    pl.DataFrame({"DFCI_MRN": ["001"], "risk_score": [2.], "outer_fold": [1]}).write_csv(path)
    result = prep._load_scores(path, "stage", "comparator")
    assert result["DFCI_MRN"].to_list() == ["001"]
    assert result["comparator_score"].to_list() == [2.]
    assert result["comparator_fold"].to_list() == [1.]


def test_cli_writes_both_figures_and_audit_with_requested_thresholds(score_tree, monkeypatch):
    from figures import io

    root, _, _ = score_tree
    output = root / "figure_data"
    monkeypatch.setattr(io, "FIGURE_DATA_DIR", str(output))
    monkeypatch.setattr(sys, "argv", ["within_cancer", "--min-patients", "3", "--min-events", "2"])
    prep.main()
    assert {path.name for path in output.iterdir()} == {
        "fig2_within_cancer_cindex.csv", "fig3_within_cancer_cindex.csv", "within_cancer_audit.csv"
    }
    for name in ("fig2_within_cancer_cindex.csv", "fig3_within_cancer_cindex.csv"):
        frame = pl.read_csv(output / name, schema_overrides={"event": pl.String})
        assert frame["event"].to_list() == ["death"]
        assert frame["min_patients_required"].to_list() == [3]
        assert frame["min_events_required"].to_list() == [2]
    audit = pl.read_csv(output / "within_cancer_audit.csv")
    assert audit.filter(pl.col("status") == "evaluated").height == 2


@pytest.mark.parametrize("min_patients,min_events", [(1, 1), (20, 0)])
def test_invalid_thresholds_fail_before_io(min_patients, min_events):
    with pytest.raises(ValueError, match="min_patients"):
        prep.prepare_within_cancer(min_patients=min_patients, min_events=min_events)
