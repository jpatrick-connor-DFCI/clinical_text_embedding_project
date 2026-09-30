import polars as pl

from figures.prep import figure2_anchor as prep
from shared.palette import MODALITY_ORDER


def _write_test_csv(path, cindex: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"mean_auc(t)": [cindex + 0.01], "mean_ibs": [0.2], "mean_c_index": [cindex]}).write_csv(path)


def test_reads_only_sequencing_os_notebook_outputs(tmp_path, monkeypatch) -> None:
    full_dir = tmp_path / "full"
    results_dir = tmp_path / "results"
    risk_dir = tmp_path / "risk"
    seen_anchors = set()

    def _full_dir(scheme, event, anchor):
        seen_anchors.add((scheme, event, anchor))
        return str(full_dir)

    def _results_dir(scheme, anchor):
        seen_anchors.add((scheme, "death", anchor))
        return str(results_dir)

    def _risk_dir(scheme, event, anchor):
        seen_anchors.add((scheme, event, anchor))
        return str(risk_dir)

    monkeypatch.setattr(prep, "full_cohort_event_dir", _full_dir)
    monkeypatch.setattr(prep, "scheme_results_dir", _results_dir)
    monkeypatch.setattr(prep, "feature_held_out_dir", _risk_dir)
    outcomes = pl.DataFrame({"DFCI_MRN": [1, 2, 3, 4, 5], "death": [1, 0, 1, 1, 0],
                             "tt_death": [10.0, 20.0, 30.0, 40.0, 50.0]})

    _write_test_csv(full_dir / "base_test.csv", 0.70)
    _write_test_csv(full_dir / "text_test.csv", 0.76)
    for i, mod in enumerate(MODALITY_ORDER):
        if mod == "prs":
            continue  # a modality whose run has not completed is left out, not NaN-filled
        _write_test_csv(results_dir / "feature_comps" / "death" / f"{mod}_test.csv", 0.6 + i / 100)
    risk_dir.mkdir()
    pl.DataFrame({"DFCI_MRN": [1, 2, 3], "risk_score": [0.1, 0.2, 0.3]}).write_csv(
        risk_dir / "text_risk_scores.csv")

    df = pl.DataFrame(
        prep._full_cohort_rows(outcomes) + prep._common_cohort_rows(outcomes),
        schema=prep._SCHEMA,
    )

    assert seen_anchors == {("death_met", "death", "sequencing")}
    assert df.filter(pl.col("cohort") == "full")["model"].to_list() == ["base", "text"]
    assert df.filter(pl.col("cohort") == "full")[["n", "n_events"]].row(0) == (5, 3)
    common = df.filter(pl.col("cohort") == "common")
    assert common["model"].to_list() == [m for m in MODALITY_ORDER if m != "prs"]
    text = common.filter(pl.col("model") == "text").row(0, named=True)
    assert (text["n"], text["n_events"]) == (3, 2)
    assert common.filter(pl.col("model") == "stage")["n"].is_null().all()
    assert df.columns == prep.ANCHOR_SENSITIVITY_COLUMNS
