"""Check the stacked modality-combination Cox models for the Figure 3 supplement."""

import numpy as np
import polars as pl
import pytest
from sksurv.metrics import concordance_index_censored

from figures.prep import figure3_combined as prep
from figures.prep.within_cancer import _block_concordance
from shared.palette import MODALITY_ORDER

# Log-hazard weight of each modality's latent signal in the synthetic outcomes.
EFFECTS = {"stage": 0.6, "treatment": 0.2, "somatic": 0.3, "prs": 0.0, "metburden": 0.4, "text": 0.8}


def _score_tables(n, seed, *, shuffled_fold_modality=None):
    """(outcomes, {modality: held-out score table}) for n synthetic patients.

    Every modality shares one fold assignment except `shuffled_fold_modality`,
    whose folds are drawn independently (so fold blocks split).
    """
    rng = np.random.default_rng(seed)
    ids = [f"P{i:05d}" for i in range(n)]
    folds = rng.permutation(np.arange(n) % 5)
    latent = {m: rng.normal(size=n) for m in MODALITY_ORDER}
    log_hazard = sum(EFFECTS[m] * latent[m] for m in MODALITY_ORDER)
    time = np.round(rng.exponential(np.exp(-log_hazard)), 2) + 0.01  # rounding creates tied times
    censor = rng.exponential(2.0, size=n)
    outcomes = pl.DataFrame({
        "DFCI_MRN": ids, "death": (time <= censor).astype(int), "tt_death": np.minimum(time, censor),
    })
    tables = {}
    for m in MODALITY_ORDER:
        fold = rng.permutation(folds) if m == shuffled_fold_modality else folds
        # Fold-specific shifts and scales mimic separately fitted outer-fold models.
        score = (latent[m] + rng.normal(scale=0.7, size=n)) * (1 + 0.2 * fold) + fold
        tables[m] = pl.DataFrame({
            "DFCI_MRN": ids, "outer_fold": fold, "selected_l1_ratio": 0.5,
            "selected_alpha": 0.01, f"{m}_risk_score": score,
        })
    return outcomes, tables


def _cohort(n, seed, **kwargs):
    outcomes, tables = _score_tables(n, seed, **kwargs)
    frame = outcomes.select(
        "DFCI_MRN", pl.col("death").cast(pl.Float64).alias("event_flag"),
        pl.col("tt_death").alias("time"),
    )
    for m, table in tables.items():
        frame = frame.join(table.select(
            "DFCI_MRN", pl.col(f"{m}_risk_score").alias(f"{m}_score"),
            pl.col("outer_fold").alias(f"{m}_fold"),
        ), on="DFCI_MRN")
    return frame


def _write_endpoint(root, scheme, event, outcomes, tables, skip=()):
    directory = root / "results" / scheme / "held_out_risk_scores" / event
    directory.mkdir(parents=True)
    for m, table in tables.items():
        if m not in skip:
            table.write_csv(directory / f"{m}_risk_scores.csv")
    outcomes.rename({"death": event, "tt_death": f"tt_{event}"}).write_parquet(
        root / f"{scheme}.parquet")


def test_models_and_contrasts_cover_the_requested_comparisons():
    non_text = [m for m in MODALITY_ORDER if m != "text"]
    assert prep.MODELS["all"] == tuple(MODALITY_ORDER)
    assert prep.MODELS["all_minus_text"] == tuple(non_text)
    assert prep.MODELS["text"] == ("text",)
    for m in non_text:
        assert prep.MODELS[m] == (m,) and prep.MODELS[f"{m}+text"] == (m, "text")
        assert (f"{m}+text", m) in prep.CONTRASTS
    assert {("all", "all_minus_text"), ("all", "text"), ("text", "all_minus_text")} <= set(prep.CONTRASTS)
    assert all(a in prep.MODELS and b in prep.MODELS for a, b in prep.CONTRASTS)


def test_pair_weighted_concordance_matches_sksurv_within_blocks():
    rng = np.random.default_rng(0)
    n = 300
    time = rng.integers(1, 40, size=n).astype(float)   # many tied times
    event = rng.uniform(size=n) < 0.6
    risks = np.vstack([rng.integers(0, 6, size=n), rng.normal(size=n)]).astype(float)  # tied risks
    blocks = rng.integers(0, 4, size=n)
    numerators, pairs, n_blocks = prep.pair_weighted_concordance(
        time, event, risks, np.ones((1, n)), blocks)

    expected_num, expected_pairs = np.zeros(2), 0
    for b in range(4):
        idx = blocks == b
        block = pl.DataFrame({"event_flag": event[idx].astype(float), "time": time[idx],
                              "a": risks[0, idx], "b": risks[1, idx]})
        num, count = _block_concordance(block, ["a", "b"])
        expected_num += num
        expected_pairs += count
    assert pairs[0] == expected_pairs and n_blocks == 4
    assert np.allclose(numerators[0], expected_num)

    # Weights act as patient multiplicities: equal to sksurv on the expanded data.
    weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    num_w, pairs_w, _ = prep.pair_weighted_concordance(
        time, event, risks, weights[None, :], np.zeros(n))
    rows = np.repeat(np.arange(n), weights.astype(int))
    for k in range(2):
        _, conc, disc, tied_risk, _ = concordance_index_censored(event[rows], time[rows], risks[k, rows])
        assert np.isclose(num_w[0, k], conc + 0.5 * tied_risk)
        assert np.isclose(pairs_w[0], conc + disc + tied_risk)


def test_cross_fitted_risk_uses_only_other_folds():
    frame = prep._standardize_within_folds(_cohort(400, 1))
    x = frame.select("stage_z", "text_z").to_numpy()
    event = frame["event_flag"].to_numpy() == 1.0
    time = frame["time"].to_numpy()
    folds = frame[prep.STACKING_FOLD].to_numpy()
    risk = prep._cross_fitted_risk(x, event, time, folds)

    held_out = folds == 0
    time_changed = np.where(held_out, time[::-1], time)
    event_changed = np.where(held_out, ~event, event)
    changed = prep._cross_fitted_risk(x, event_changed, time_changed, folds)
    assert np.allclose(risk[held_out], changed[held_out])     # fold 0's own outcomes are unused
    assert not np.allclose(risk[~held_out], changed[~held_out])

    assert prep._cross_fitted_risk(np.ones((400, 1)), event, time, folds) is None


def test_single_score_model_reproduces_the_raw_score_cindex():
    # With a positive coefficient in every fold, stacking one standardized score is
    # monotone within each fold block, so the C-index is the raw score's.
    frame = _cohort(600, 2)
    results, _, _ = prep.evaluate_endpoint(frame, scheme="death_met", event="death")
    text = results.filter(pl.col("model") == "text").row(0, named=True)
    # Every modality shares one fold assignment here, so the fold blocks are the folds.
    blocks = [_block_concordance(block, ["text_score"])
              for block in frame.partition_by("text_fold")]
    num, count = sum(n[0] for n, _ in blocks), sum(c for _, c in blocks)
    assert text["status"] == "ok" and text["n_fold_blocks"] == 5
    assert text["n_comparable_pairs"] == count
    assert text["cindex"] == pytest.approx(num / count)


def test_evaluate_endpoint_reports_every_model_and_bootstrap_contrasts():
    frame = _cohort(800, 3, shuffled_fold_modality="prs")
    results, os_cindex, os_delta = prep.evaluate_endpoint(
        frame, scheme="death_met", event="death", n_boot=200)
    assert results.columns == list(prep.CINDEX_SCHEMA)
    assert results["model"].to_list() == list(prep.MODELS)
    assert set(results["status"]) == {"ok"}
    assert results["n_patients"].unique().to_list() == [800]
    assert results["n_comparable_pairs"].n_unique() == 1   # one shared set of pairs
    assert results["n_fold_blocks"][0] > 5                 # prs folds split the blocks
    cindex = dict(zip(results["model"], results["cindex"]))
    assert cindex["stage+text"] > cindex["stage"] and cindex["prs+text"] > cindex["prs"]
    assert cindex["all"] > cindex["all_minus_text"] and cindex["all"] > cindex["text"]

    assert os_cindex["model"].to_list() == list(prep.MODELS)
    assert (os_cindex["n_boot"] == 200).all()
    assert ((os_cindex["ci_lower"] <= os_cindex["cindex"])
            & (os_cindex["cindex"] <= os_cindex["ci_upper"])).all()
    assert list(zip(os_delta["model"], os_delta["reference"])) == list(prep.CONTRASTS)
    for row in os_delta.iter_rows(named=True):
        assert row["delta_cindex"] == pytest.approx(cindex[row["model"]] - cindex[row["reference"]])
        assert row["ci_lower"] <= row["delta_cindex"] <= row["ci_upper"]
    stage_text = os_delta.filter(pl.col("model") == "stage+text").row(0, named=True)
    assert stage_text["ci_lower"] > 0


@pytest.mark.parametrize(("n", "event_rate", "status"), [
    (15, 0.5, "too_few_patients"), (100, 0.02, "too_few_events"), (100, 0.98, "too_few_non_events"),
])
def test_ineligible_endpoints_get_a_status_for_every_model(n, event_rate, status):
    frame = _cohort(n, 4)
    events = np.zeros(n)
    events[: max(1, round(event_rate * n))] = 1.0
    results, os_cindex, os_delta = prep.evaluate_endpoint(
        frame.with_columns(pl.Series("event_flag", events)), scheme="s", event="e", n_boot=10)
    assert results["model"].to_list() == list(prep.MODELS)
    assert set(results["status"]) == {status} and results["cindex"].is_null().all()
    assert os_cindex.is_empty() and os_delta.is_empty()


def test_prepare_combined_reads_endpoints_and_matches_serial(tmp_path, monkeypatch):
    outcomes, tables = _score_tables(300, 5)
    _write_endpoint(tmp_path, "death_met", "death", outcomes, tables)
    other = tmp_path / "results" / "death_met" / "held_out_risk_scores" / "brain"
    other.mkdir()
    for m, table in tables.items():
        if m != "somatic":
            table.write_csv(other / f"{m}_risk_scores.csv")
    monkeypatch.setattr(prep, "SCHEMES", ["death_met", "icd3_post"])
    monkeypatch.setattr(prep, "SURV_PATH", str(tmp_path))
    monkeypatch.setattr(prep, "embedding_file", lambda scheme: f"{scheme}.parquet")
    monkeypatch.setattr(prep, "scheme_results_dir", lambda scheme: str(tmp_path / "results" / scheme))

    tasks = prep._endpoint_tasks(n_boot=50)
    assert [(t[0], t[1], t[4]) for t in tasks] == [("death_met", "brain", 0), ("death_met", "death", 50)]

    serial = prep.prepare_combined(n_boot=50, n_jobs=1, show_progress=False)
    cindex, os_cindex, os_delta = serial
    assert set(cindex.filter(pl.col("event") == "brain")["status"]) == {"missing_inputs"}
    assert set(cindex.filter(pl.col("event") == "death")["status"]) == {"ok"}
    assert os_cindex.height == len(prep.MODELS) and os_delta.height == len(prep.CONTRASTS)

    parallel = prep.prepare_combined(n_boot=50, n_jobs=2, show_progress=False)
    for expected, actual in zip(serial, parallel):
        assert actual.equals(expected)
