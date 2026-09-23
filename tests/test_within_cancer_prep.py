"""Check paired, fold-aware survival comparisons and their source-data contracts."""

import gzip
import sys
import warnings

import numpy as np
import polars as pl
import pytest

from figures.prep import within_cancer as prep


FEATURE_MEMBERSHIP_FILES = (
    "complete_somatic_data_df.csv.gz",
    "complete_germline_data_df.csv.gz",
    "cancer_stage_df.csv.gz",
    "categorical_treatment_data_by_line.csv.gz",
)


def _write_gzip_csv(path, frame):
    with gzip.open(path, "wt") as handle:
        handle.write(frame.write_csv())


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
    if status != "ok":
        assert row["n_comparable_pairs"] is None
        assert row["n_fold_blocks"] is None


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
    for name in FEATURE_MEMBERSHIP_FILES:
        _write_gzip_csv(tmp_path / name, pl.DataFrame({"DFCI_MRN": ids}))
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
    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer(min_patients=2, min_events=1)
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
    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert fig2.is_empty()
    assert not fig3.is_empty()
    assert not caplog.text
    assert audit["detail"].str.contains("missing outer_fold").any()
    assert audit.filter(pl.col("status") == "invalid_text_or_outcome_input").height == 1


def test_duplicate_predictions_skip_comparator_and_record_reason(score_tree):
    _, _, features = score_tree
    path = features / "stage_risk_scores.csv"
    scores = pl.read_csv(path)
    pl.concat([scores, scores.head(1)]).write_csv(path)
    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert not fig2.is_empty()
    assert fig3.is_empty()
    assert audit.filter(pl.col("detail").str.contains("duplicate patient IDs")).height == 1


def test_missing_inputs_preserve_typed_schemas_and_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "FEATURE_PATH", str(tmp_path))
    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer()
    assert fig2.is_empty() and fig3.is_empty()
    assert fig2.schema == fig3.schema == pl.Schema(prep.RESULT_SCHEMA)
    assert audit.schema == pl.Schema(prep.AUDIT_SCHEMA)
    assert counts2.schema == counts3.schema == pl.Schema(prep.COUNT_SCHEMA)
    assert counts2.is_empty() and counts3.is_empty()
    assert audit["status"].to_list() == ["invalid_cancer_input"]


def test_missing_raw_cancer_label_cannot_fall_back_to_dummies(score_tree):
    root, _, _ = score_tree
    path = root / "cancer_type_df.csv.gz"
    labels = pl.read_csv(path).drop("CANCER_TYPE")
    with gzip.open(path, "wt") as handle:
        handle.write(labels.write_csv())
    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer(min_patients=2, min_events=1)
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


def test_cli_writes_both_figures_counts_and_audit_quietly(score_tree, monkeypatch, capsys, caplog):
    root, _, _ = score_tree
    output = root / "figure_data"
    monkeypatch.setattr(prep, "FIGURE_DATA_DIR", str(output))
    monkeypatch.setattr(sys, "argv", ["within_cancer", "--min-patients", "3", "--min-events", "2"])
    prep.main()
    assert {path.name for path in output.iterdir()} == {
        "fig2_within_cancer_cindex.csv", "fig3_within_cancer_cindex.csv",
        "fig2_within_cancer_event_counts.csv", "fig3_within_cancer_event_counts.csv",
        "within_cancer_audit.csv", "fig3_within_cancer_modality_cindex.csv",
    }
    for name in (
        "fig2_within_cancer_cindex.csv", "fig3_within_cancer_cindex.csv",
        "fig2_within_cancer_event_counts.csv", "fig3_within_cancer_event_counts.csv",
    ):
        frame = pl.read_csv(output / name, schema_overrides={"event": pl.String})
        assert frame["event"].to_list() == ["death"]
        assert frame["min_patients_required"].to_list() == [3]
        assert frame["min_events_required"].to_list() == [2]
    audit = pl.read_csv(output / "within_cancer_audit.csv")
    assert audit.filter(pl.col("status") == "evaluated").height == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[wrote]" not in captured.err and "WARNING" not in captured.err
    assert not caplog.text


@pytest.mark.parametrize("min_patients,min_events", [(1, 1), (20, 0)])
def test_invalid_thresholds_fail_before_io(min_patients, min_events):
    with pytest.raises(ValueError, match="min_patients"):
        prep.prepare_within_cancer(min_patients=min_patients, min_events=min_events)


def test_source_counts_precede_predictions_and_use_all_common_feature_ids(score_tree):
    root, full, features = score_tree
    ids = [str(i) for i in range(6)]
    # Every modality file excludes a different patient. Only 0 and 1 occur in all four.
    for index, name in enumerate(FEATURE_MEMBERSHIP_FILES, start=2):
        _write_gzip_csv(root / name, pl.DataFrame({
            "DFCI_MRN": [patient for patient in ids if patient != str(index)]
        }))
    # Metastatic burden is left-joined/zero-filled in training, so even a stale
    # membership file must not reduce the common modality population.
    _write_gzip_csv(root / "met_burden_df.csv.gz", pl.DataFrame({"DFCI_MRN": ["absent"]}))
    for path in full.glob("*_risk_scores.csv"):
        pl.read_csv(path).head(4).write_csv(path)
    for path in features.glob("*_risk_scores.csv"):
        pl.read_csv(path).head(2).write_csv(path)

    fig2, fig3, _, counts2, counts3, _ = prep.prepare_within_cancer(
        min_patients=2, min_events=1, show_progress=False
    )
    assert counts2.schema == counts3.schema == pl.Schema(prep.COUNT_SCHEMA)
    assert counts2.select("n_patients", "n_events", "n_non_events").rows() == [(6, 6, 0)]
    assert counts3.select("n_patients", "n_events", "n_non_events").rows() == [(2, 2, 0)]
    assert counts2["eligible"].to_list() == counts3["eligible"].to_list() == [True]
    assert fig2["n_patients"].to_list() == [4]
    assert fig3["n_patients"].to_list() == [2]


def test_untrained_endpoints_count_valid_outcomes_and_preserve_zero_strata(score_tree):
    root, _, _ = score_tree
    label_path = root / "cancer_type_df.csv.gz"
    labels = pl.read_csv(label_path).with_columns(
        pl.Series("CANCER_TYPE", ["Breast"] * 3 + ["Lung"] * 3)
    )
    _write_gzip_csv(label_path, labels)
    outcome_path = root / prep.embedding_file("death_met")
    outcomes = pl.read_parquet(outcome_path).with_columns(
        pl.Series("untrained", [1., 0., 2., np.nan, 1., 0.]),
        pl.Series("tt_untrained", [1., 2., 3., 4., 0., np.inf]),
    )
    outcomes.write_parquet(outcome_path)

    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer(
        min_patients=2, min_events=1, show_progress=False
    )
    for counts in (counts2, counts3):
        untrained = counts.filter(pl.col("event") == "untrained").sort("cancer_type")
        assert untrained.select("cancer_type", "n_patients", "n_events", "n_non_events").rows() == [
            ("Breast", 2, 1, 1), ("Lung", 0, 0, 0)
        ]
        assert untrained["eligible"].to_list() == [True, False]
        assert untrained["status"][1] == "too_few_patients"
    assert fig2["event"].unique().to_list() == fig3["event"].unique().to_list() == ["death"]
    assert audit.filter(pl.col("event") == "untrained").height > 0


def test_cancers_absent_from_shared_modality_cohort_have_zero_counts(score_tree):
    root, _, _ = score_tree
    labels_path = root / "cancer_type_df.csv.gz"
    labels = pl.read_csv(labels_path).with_columns(
        pl.Series("CANCER_TYPE", ["Breast"] * 3 + ["Lung"] * 3)
    )
    _write_gzip_csv(labels_path, labels)
    for name in FEATURE_MEMBERSHIP_FILES:
        _write_gzip_csv(root / name, pl.DataFrame({"DFCI_MRN": ["0", "1", "2"]}))
    _, _, _, _, counts3, _ = prep.prepare_within_cancer(
        min_patients=2, min_events=1, show_progress=False
    )
    assert counts3.select("cancer_type", "n_patients", "n_events", "eligible").rows() == [
        ("Breast", 3, 3, True), ("Lung", 0, 0, False)
    ]


def test_brain_metastasis_counts_follow_primary_brain_exclusion(score_tree):
    root, _, _ = score_tree
    label_path = root / "cancer_type_df.csv.gz"
    labels = pl.read_csv(label_path).with_columns(
        pl.Series("CANCER_TYPE", ["BRAIN", "BRAIN"] + ["Reference cancer"] * 4),
        pl.Series("CANCER_TYPE_BRAIN", [1., 1., 0., 0., np.nan, None]),
    )
    _write_gzip_csv(label_path, labels)
    outcome_path = root / prep.embedding_file("death_met")
    pl.read_parquet(outcome_path).with_columns(
        pl.lit(1).alias("brainM"), pl.lit(1.).alias("tt_brainM")
    ).write_parquet(outcome_path)

    _, _, _, counts2, counts3, _ = prep.prepare_within_cancer(
        min_patients=2, min_events=1, show_progress=False
    )
    for counts in (counts2, counts3):
        brain = counts.filter(pl.col("event") == "brainM").sort("cancer_type")
        assert brain.select("cancer_type", "n_patients", "n_events").rows() == [
            ("BRAIN", 0, 0), ("Reference cancer", 4, 4)
        ]
        # Other endpoints must keep the primary brain patients.
        assert counts.filter((pl.col("event") == "death") & (pl.col("cancer_type") == "BRAIN"))["n_patients"].to_list() == [2]


def test_precounts_skip_all_prediction_and_concordance_work_when_ineligible(score_tree, monkeypatch):
    def unexpected_work(*args, **kwargs):
        pytest.fail("Ineligible source cohorts must be screened before prediction reads or concordance")

    monkeypatch.setattr(prep, "_load_scores", unexpected_work)
    monkeypatch.setattr(prep, "_block_pair_counts", unexpected_work)
    _, _, _, counts2, counts3, _ = prep.prepare_within_cancer(
        min_patients=7, min_events=1, show_progress=False
    )
    for counts in (counts2, counts3):
        assert counts["n_patients"].to_list() == [6]
        assert counts["eligible"].to_list() == [False]
        assert counts["status"].to_list() == ["too_few_patients"]


@pytest.mark.parametrize("min_patients,min_events", [(5, 1), (2, 3)])
def test_sparse_matched_strata_do_not_compute_concordance(monkeypatch, min_patients, min_events):
    def unexpected_concordance(*args, **kwargs):
        pytest.fail("Insufficient matched patient/event counts must skip concordance")

    monkeypatch.setattr(prep, "_block_pair_counts", unexpected_concordance)
    frame = _paired([1., 2., 3., 4.], [1., 0., 1., 0.], [4., 3., 2., 1.], [1., 2., 3., 4.])
    row = _evaluate(frame, min_patients=min_patients, min_events=min_events).row(0, named=True)
    assert row["n_comparable_pairs"] is None
    assert row["n_fold_blocks"] is None
    assert row["text_cindex"] is None


def test_matched_counts_rechecked_after_source_counts_pass(score_tree, monkeypatch):
    original = prep._block_pair_counts
    evaluated_sizes = []

    def record_concordance(block):
        evaluated_sizes.append(block.height)
        return original(block)

    monkeypatch.setattr(prep, "_block_pair_counts", record_concordance)
    fig2, fig3, _, counts2, counts3, _ = prep.prepare_within_cancer(
        min_patients=6, min_events=1, show_progress=False
    )
    assert counts2["eligible"].to_list() == counts3["eligible"].to_list() == [True]
    assert fig2["status"].to_list() == ["ok"]
    assert fig3["n_patients"].to_list() == [5]
    assert fig3["status"].to_list() == ["too_few_patients"]
    assert fig3["n_comparable_pairs"].to_list() == fig3["n_fold_blocks"].to_list() == [None]
    assert evaluated_sizes == [6]


def test_missing_modality_membership_preserves_score_evaluation_and_audits_counts(score_tree):
    root, _, _ = score_tree
    (root / FEATURE_MEMBERSHIP_FILES[0]).unlink()
    fig2, fig3, audit, counts2, counts3, modality = prep.prepare_within_cancer(
        min_patients=2, min_events=1, show_progress=False
    )
    assert not fig2.is_empty() and not fig3.is_empty()
    assert not counts2.is_empty() and counts3.is_empty()
    assert counts3.schema == pl.Schema(prep.COUNT_SCHEMA)
    assert audit["detail"].str.contains(FEATURE_MEMBERSHIP_FILES[0], literal=True).any()


def test_warnings_suppressed_and_progress_has_two_global_endpoint_bars(score_tree, monkeypatch, caplog):
    root, _, _ = score_tree
    monkeypatch.setattr(prep, "SCHEMES", ["death_met", "icd3_post"])
    first_path = root / prep.embedding_file("death_met")
    first = pl.read_parquet(first_path)
    first.write_parquet(root / prep.embedding_file("icd3_post"))
    first.with_columns(pl.lit(1).alias("untrained"), pl.lit(2.).alias("tt_untrained")).write_parquet(first_path)
    bars = []

    class ProgressSpy:
        def __init__(self, iterable=None, *, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total if total is not None else len(iterable)
            self.desc = desc
            self.n = 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update()

        def update(self, n=1):
            self.n += n

        def close(self):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass

    original = prep._read_outcomes

    def noisy_read(*args, **kwargs):
        warnings.warn("synthetic endpoint warning", RuntimeWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(prep, "tqdm", ProgressSpy)
    monkeypatch.setattr(prep, "_read_outcomes", noisy_read)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert not emitted
    assert not caplog.text
    assert [(bar.desc, bar.total, bar.n) for bar in bars] == [
        ("Full cohort", 3, 3), ("Modality cohort", 3, 3)
    ]
    # The preparation call must not leave Python warnings disabled for its caller.
    with warnings.catch_warnings(record=True) as emitted_after:
        warnings.simplefilter("always")
        warnings.warn("caller warning", RuntimeWarning)
    assert len(emitted_after) == 1


def _modality_frame(times, events, scores, folds):
    n = len(times)
    data = {"DFCI_MRN": [str(i) for i in range(n)], "time": times, "event_flag": events,
            "cancer_type": ["Reference cancer"] * n}
    for modality, values in scores.items():
        data[f"{modality}_score"] = values
        data[f"{modality}_fold"] = folds[modality]
    return pl.DataFrame(data)


def test_all_modalities_share_patients_pairs_and_joint_fold_blocks():
    # Only patients 0-1 and 2-3 share every model's fold, so every modality is
    # scored on exactly those two pairs even though "c" used a single fold.
    frame = _modality_frame(
        [1., 2., 3., 4.], [1.] * 4,
        {"a": [4., 3., 2., 1.], "b": [1., 2., 3., 4.], "c": [2., 1., 4., 3.]},
        {"a": [0, 0, 1, 1], "b": [0, 0, 1, 1], "c": [0, 0, 0, 0]},
    )
    result = prep.evaluate_modalities(frame, scheme="s", event="e", modalities=["a", "b", "c"],
                                      min_patients=2, min_events=1)
    by_modality = dict(zip(result["modality"], result["cindex"]))
    assert by_modality == {"a": 1.0, "b": 0.0, "c": 1.0}
    assert set(result["n_comparable_pairs"]) == {2}
    assert set(result["n_fold_blocks"]) == {2}
    # With two modalities, this matches the paired evaluation of the same scores.
    pairwise = _evaluate(_paired([1., 2., 3., 4.], [1.] * 4, [4., 3., 2., 1.], [1., 2., 3., 4.],
                                 text_folds=[0, 0, 1, 1], other_folds=[0, 0, 1, 1])).row(0, named=True)
    assert (pairwise["text_cindex"], pairwise["comparator_cindex"]) == (1.0, 0.0)


def test_all_modality_table_requires_every_modality(score_tree):
    _, _, features = score_tree
    *_, audit, _, _, modality = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert modality.is_empty() and modality.schema == pl.Schema(prep.MODALITY_RESULT_SCHEMA)
    assert audit.filter(pl.col("status") == "incomplete_modalities").height == 1

    ids = [str(i) for i in range(6)]
    for name in set(prep.MODALITY_ORDER) - {"text", "stage"}:
        pl.DataFrame({"DFCI_MRN": ids, f"{name}_risk_score": [1.] * 6, "outer_fold": [0] * 6}
                     ).write_csv(features / f"{name}_risk_scores.csv")
    *_, modality = prep.prepare_within_cancer(min_patients=2, min_events=1)
    assert set(modality["modality"]) == set(prep.MODALITY_ORDER)
    # Stage covers five patients, so every modality is evaluated on those five.
    assert set(modality["n_patients"]) == {5}
    cindex = dict(zip(modality["modality"], modality["cindex"]))
    assert cindex["stage"] == 1.0 and cindex["text"] == 0.0 and cindex["somatic"] == 0.5


def test_parallel_endpoints_match_serial(score_tree):
    tmp_path, full, features = score_tree
    # A second endpoint so the pool has more than one task per comparison.
    path = tmp_path / prep.embedding_file("death_met")
    pl.read_parquet(path).with_columns(
        pl.Series("progression", [1, 0, 1, 1, 0, 1]), pl.Series("tt_progression", [2., 1., 4., 3., 6., 5.]),
    ).write_parquet(path)
    for root in (full, features):
        target = root.parent / "progression"
        target.mkdir()
        for file in root.iterdir():
            (target / file.name).write_bytes(file.read_bytes())
    serial = prep.prepare_within_cancer(min_patients=2, min_events=1, n_jobs=1)
    parallel = prep.prepare_within_cancer(min_patients=2, min_events=1, n_jobs=2)
    assert set(serial[0]["event"]) == {"death", "progression"}
    for expected, actual in zip(serial, parallel):
        assert actual.equals(expected)
