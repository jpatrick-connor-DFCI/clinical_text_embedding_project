"""Check the published-score-vs-text C-index/Cox/KM prep, modeled on
test_figure3_combined_prep.py. Uses the real mgps PublishedScore (integer
points 0-2, direction=1) so ties-on-published behave like the real catalog."""

import numpy as np
import polars as pl
import pytest

from figures.prep import published_scores as prep
from figures.prep.within_cancer import _block_concordance
from shared.published_scores import default_catalog

SCORE = default_catalog()["mgps"]


def _cohort(n, seed, *, text_weight=1.0, n_folds=5):
    """Synthetic score cohort: published points (0-2, heavily tied) plus a
    text score correlated with the same latent risk, so published+text beats
    published alone. Columns match what evaluate_score/_cox/_km/_cohort need."""
    rng = np.random.default_rng(seed)
    ids = [f"P{i:05d}" for i in range(n)]
    latent = rng.normal(size=n)
    points = np.clip(np.round(latent + rng.normal(scale=0.5, size=n)), 0, 2)
    text_fold = rng.integers(0, n_folds, size=n)
    text_score = (
        text_weight * latent + rng.normal(scale=0.6, size=n)
    ) * (1 + 0.1 * text_fold) + text_fold
    log_hazard = 0.5 * points + 0.7 * latent
    time = np.round(rng.exponential(np.exp(-log_hazard)), 2) + 0.01
    censor = rng.exponential(2.0, size=n)
    event = (time <= censor).astype(float)
    frame = pl.DataFrame({
        "DFCI_MRN": ids,
        "published_score": points,
        "published_group": [str(int(p)) for p in points],
        "labs_observable": True,
        "text_score": text_score,
        "text_fold": text_fold.astype(float),
        "event_flag": event,
        "time": np.minimum(time, censor),
    })
    return frame


def test_score_cohort_filters_to_eligible_observable_complete_scored():
    score_df = pl.DataFrame({
        "DFCI_MRN": ["P1", "P2", "P3", "P4", "P5"],
        "mgps__eligible": [True, True, False, True, True],
        "mgps__complete": [True, False, True, True, True],
        "mgps__points": [0.0, 1.0, 2.0, 1.0, None],
        "mgps__group": ["0", "1", "2", "1", None],
        "labs_observable": [True, True, True, False, True],
    })
    text = pl.DataFrame({
        "DFCI_MRN": ["P1", "P2", "P3", "P4", "P5"],
        "text_score": [0.1, 0.2, 0.3, 0.4, 0.5],
        prep.TEXT_FOLD: [0.0, 1.0, 2.0, 3.0, 4.0],
    })
    outcomes = pl.DataFrame({
        "DFCI_MRN": ["P1", "P2", "P3", "P4", "P5"],
        "event_flag": [1.0, 1.0, 1.0, 1.0, 1.0],
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    frame = prep._score_cohort(score_df, text, outcomes, SCORE)
    # Only P1 is eligible, complete, observable and scored (P5 has null points too).
    assert frame["DFCI_MRN"].to_list() == ["P1"]



def test_score_cohort_handles_all_null_points_read_back_from_csv(tmp_path):
    """An all-null points/group column comes back from CSV as String; that used
    to raise inside _score_cohort and get mislabeled missing_inputs."""
    path = tmp_path / "published_scores_df.csv.gz"
    pl.DataFrame({
        "DFCI_MRN": ["P1", "P2"],
        "mgps__eligible": [True, True],
        "mgps__complete": [False, False],
        "mgps__points": [None, None],
        "mgps__group": [None, None],
        "labs_observable": [True, True],
    }, schema_overrides={"mgps__points": pl.Float64, "mgps__group": pl.String}).write_csv(path, compression="gzip")
    score_df = pl.read_csv(path, schema_overrides={"DFCI_MRN": pl.String})
    assert score_df.schema["mgps__points"] == pl.String
    text = pl.DataFrame({"DFCI_MRN": ["P1", "P2"], "text_score": [0.1, 0.2], prep.TEXT_FOLD: [0.0, 1.0]})
    outcomes = pl.DataFrame({"DFCI_MRN": ["P1", "P2"], "event_flag": [1.0, 0.0], "time": [1.0, 2.0]})
    assert prep._score_cohort(score_df, text, outcomes, SCORE).is_empty()

def test_evaluate_score_reports_every_model_and_bootstrap_contrasts():
    frame = _cohort(600, 1)
    cindex, delta = prep.evaluate_score(
        frame, anchor="treatment", lab_window_days=30, score=SCORE, n_boot=200)
    assert cindex.columns == list(prep.CINDEX_SCHEMA)
    assert cindex["model"].to_list() == list(prep.MODELS)
    assert set(cindex["status"]) == {"ok"}
    assert cindex["n_patients"].unique().to_list() == [600]

    c = dict(zip(cindex["model"], cindex["cindex"]))
    assert c["published+text"] > c["published"]

    assert ((cindex["ci_lower"] <= cindex["cindex"]) & (cindex["cindex"] <= cindex["ci_upper"])).all()
    assert list(zip(delta["model"], delta["reference"])) == list(prep.CONTRASTS)
    for row in delta.iter_rows(named=True):
        assert row["delta_cindex"] == pytest.approx(c[row["model"]] - c[row["reference"]])
        assert row["ci_lower"] <= row["delta_cindex"] <= row["ci_upper"]
    combo_vs_pub = delta.filter(
        (pl.col("model") == "published+text") & (pl.col("reference") == "published")
    ).row(0, named=True)
    assert combo_vs_pub["ci_lower"] > 0


def test_published_model_cindex_matches_direct_pair_weighted_concordance():
    frame = _cohort(500, 2)
    cindex, _ = prep.evaluate_score(frame, anchor="treatment", lab_window_days=30, score=SCORE)
    published = cindex.filter(pl.col("model") == "published").row(0, named=True)

    blocks = [_block_concordance(block.rename({"published_score": "p"}), ["p"])
              for block in frame.with_columns(
                  (pl.col("published_score") * SCORE.direction).alias("published_score"),
              ).partition_by("text_fold")]
    num, count = sum(n[0] for n, _ in blocks), sum(c for _, c in blocks)
    assert published["n_comparable_pairs"] == count
    assert published["cindex"] == pytest.approx(num / count)


def test_serial_and_parallel_runs_match(tmp_path, monkeypatch):
    frame = _cohort(300, 3)
    score_df = frame.select(
        "DFCI_MRN",
        pl.lit(True).alias("mgps__eligible"),
        pl.lit(True).alias("mgps__complete"),
        pl.col("published_score").alias("mgps__points"),
        pl.col("published_group").alias("mgps__group"),
        "labs_observable",
    )
    text = frame.select("DFCI_MRN", "text_score", pl.col("text_fold").alias(prep.TEXT_FOLD))
    outcomes = frame.select("DFCI_MRN", "event_flag", "time")

    score_df.write_csv(tmp_path / "published_scores_df.csv.gz", compression="gzip")
    monkeypatch.setattr(prep, "FEATURE_PATH", str(tmp_path))
    monkeypatch.setattr(prep, "_load_text_scores", lambda: text)
    monkeypatch.setattr(prep, "_load_outcomes", lambda: outcomes)
    monkeypatch.setattr(prep, "CATALOG", {"mgps": SCORE})
    monkeypatch.setattr(prep, "ANCHORS_TO_RUN", ("treatment",))
    monkeypatch.setattr(prep, "LAB_WINDOWS_TO_RUN", (30,))

    serial = prep.prepare_published_scores(n_boot=50, n_jobs=1, show_progress=False)
    parallel = prep.prepare_published_scores(n_boot=50, n_jobs=2, show_progress=False)
    for expected, actual in zip(serial, parallel):
        assert actual.equals(expected)
    cindex = serial[0]
    assert set(cindex["status"]) == {"ok"}


@pytest.mark.parametrize(("n", "event_rate", "status"), [
    (15, 0.5, "too_few_patients"), (100, 0.02, "too_few_events"), (100, 0.98, "too_few_non_events"),
])
def test_ineligible_cohorts_get_a_status_for_every_model(n, event_rate, status):
    frame = _cohort(n, 4)
    events = np.zeros(n)
    events[: max(1, round(event_rate * n))] = 1.0
    frame = frame.with_columns(pl.Series("event_flag", events))
    cindex, delta = prep.evaluate_score(frame, anchor="treatment", lab_window_days=30, score=SCORE, n_boot=10)
    assert cindex["model"].to_list() == list(prep.MODELS)
    assert set(cindex["status"]) == {status} and cindex["cindex"].is_null().all()
    assert delta.is_empty()


def test_constant_published_status():
    frame = _cohort(200, 5).with_columns(pl.lit(1.0).alias("published_score"))
    cindex, delta = prep.evaluate_score(frame, anchor="treatment", lab_window_days=30, score=SCORE, n_boot=10)
    assert set(cindex["status"]) == {"constant_published"}
    assert delta.is_empty()


def test_changing_one_folds_outcomes_leaves_other_folds_predictions_unchanged():
    frame = _cohort(400, 6)
    frame = prep._standardize_within_folds(frame.rename({"text_score": "text_score_raw"}).with_columns(
        pl.col("text_score_raw").alias("text_score")
    ), modalities=("text",))
    x = frame.select(
        (pl.col("published_score") * SCORE.direction).alias("published_oriented"), "text_z",
    ).to_numpy()
    event = frame["event_flag"].to_numpy() == 1.0
    time = frame["time"].to_numpy()
    folds = frame[prep.TEXT_FOLD].to_numpy()
    risk = prep._cross_fitted_risk(x, event, time, folds)

    held_out = folds == 0
    time_changed = np.where(held_out, time[::-1], time)
    event_changed = np.where(held_out, ~event, event)
    changed = prep._cross_fitted_risk(x, event_changed, time_changed, folds)
    assert np.allclose(risk[held_out], changed[held_out])
    assert not np.allclose(risk[~held_out], changed[~held_out])


def test_pure_noise_text_gives_a_large_lrt_p():
    rng = np.random.default_rng(7)
    n = 500
    points = rng.integers(0, 3, size=n).astype(float)
    time = rng.exponential(1.0, size=n)
    event = (rng.uniform(size=n) < 0.6).astype(float)
    frame = pl.DataFrame({
        "DFCI_MRN": [f"P{i:05d}" for i in range(n)],
        "published_score": points,
        "text_score": rng.normal(size=n),  # pure noise, independent of outcome
        "event_flag": event,
        "time": time,
    })
    cox = prep.evaluate_cox(frame, anchor="treatment", lab_window_days=30, score=SCORE)
    text_row = cox.filter(pl.col("term") == "text_z").row(0, named=True)
    assert text_row["lrt_p"] > 0.05


def test_evaluate_km_matches_group_sizes_and_schema():
    frame = _cohort(300, 8)
    km = prep.evaluate_km(frame, anchor="treatment", lab_window_days=30, score=SCORE)
    assert km.columns == list(prep.KM_SCHEMA)
    # pl.lit(30) alone is Int32; a dtype mismatch broke the cross-task concat on the cluster.
    assert km.schema["lab_window_days"] == prep.KM_SCHEMA["lab_window_days"]
    assert km.height == frame.height
    assert set(km["text_tertile"].unique()) <= {"low", "mid", "high"}
    assert set(km["text_group_matched"].unique()) <= {g.label for g in SCORE.risk_groups}


def test_evaluate_cohort_reports_spearman_and_unblocked_cindex():
    frame = _cohort(400, 9)
    score_df = frame.select(
        "DFCI_MRN",
        pl.lit(True).alias("mgps__eligible"),
        pl.lit(True).alias("mgps__complete"),
        "labs_observable",
    )
    cohort = prep.evaluate_cohort(frame, score_df, anchor="treatment", lab_window_days=30, score=SCORE)
    row = cohort.row(0, named=True)
    assert row["n_eligible"] == 400
    assert row["frac_tied_published"] > 0  # integer points tie heavily
    assert -1.0 <= row["spearman_published_text"] <= 1.0
    assert 0.0 <= row["unblocked_published_cindex"] <= 1.0
    assert row["underpowered"] is False
