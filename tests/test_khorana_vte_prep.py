"""figures/prep/khorana_vte.py: horizon censoring and the per-run outputs,
on synthetic builder output, text scores and outcomes."""

import numpy as np
import polars as pl

from figures.prep import khorana_vte as kv
from figures.prep import published_scores as prep


def test_apply_horizon_censors_events_after_the_horizon():
    outcomes = pl.DataFrame({"DFCI_MRN": ["a", "b", "c"], "event_flag": [1.0, 1.0, 0.0], "time": [100.0, 300.0, 50.0]})
    assert kv.apply_horizon(outcomes, None).equals(outcomes)
    out = kv.apply_horizon(outcomes, 180)
    assert out["event_flag"].to_list() == [1.0, 0.0, 0.0]
    assert out["time"].to_list() == [100.0, 180.0, 50.0]


def test_evaluate_task_tags_event_and_horizon(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 400
    ids = [f"P{i}" for i in range(n)]
    points = rng.integers(0, 5, size=n).astype(float)
    latent = points + rng.normal(size=n)
    time = rng.exponential(np.exp(-0.4 * latent)) * 200 + 1
    pl.DataFrame({
        "DFCI_MRN": ids, "khorana__eligible": True, "khorana__complete": True, "khorana__points": points,
        "khorana__group": ["x"] * n, "labs_observable": True,
    }).write_csv(tmp_path / "published_scores_df.csv.gz", compression="gzip")
    text = pl.DataFrame({"DFCI_MRN": ids, "text_score": latent + rng.normal(size=n),
                         prep.TEXT_FOLD: rng.integers(0, 5, size=n).astype(float)})
    outcomes = pl.DataFrame({"DFCI_MRN": ids, "event_flag": (time < 300).astype(float), "time": np.minimum(time, 300)})
    calls = []
    monkeypatch.setattr(prep, "FEATURE_PATH", str(tmp_path))
    monkeypatch.setattr(kv, "_load_text_scores", lambda anchor, endpoint: calls.append(endpoint) or text)
    monkeypatch.setattr(kv, "_load_outcomes", lambda anchor, endpoint: outcomes)

    cindex, delta, cox, km, cohort = kv._evaluate_task(("treatment", 30, "vte_phecode", 180, 0))
    assert calls == [("vte", "vte_phecode")]
    for frame, schema in zip((cindex, delta, cox, km, cohort), kv.SCHEMAS):
        assert frame.schema == pl.Schema(schema)
    assert set(cindex["status"]) == {"ok"}
    assert cindex["event"].unique().to_list() == ["vte_phecode"]
    assert cindex["horizon_days"].unique().to_list() == [180]
    assert km["time"].max() <= 180

    missing = kv._evaluate_task(("treatment", 90, "vte", None, 0))[0]
    assert set(missing["status"]) == {"missing_inputs"}


def test_bootstrap_only_in_primary_runs():
    tasks = kv._tasks(100)
    boot = {t[:4] for t in tasks if t[4]}
    assert boot == {("treatment", 30, "vte", None), ("treatment", 30, "vte", 180)}
    assert len(tasks) == 16
