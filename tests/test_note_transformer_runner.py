"""Checkpoint selection and Cox-comparison semantics of the note-transformer runner."""

import itertools
import json
import os
import sys

import numpy as np
import polars as pl
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipelines.note_transformer import run_os_transformer as runner  # noqa: E402
from pipelines.note_transformer.note_survival_transformer.evaluate import safe_c_index  # noqa: E402


# --------------------------------------------------------------------------- oracles


def brute_pairs_and_credit(time, event, score):
    """Harrell pairs with lifelines tie rules: (n_pairs, concordant + 0.5 * tied_risk)."""
    pairs = credit = 0.0
    for i, j in itertools.permutations(range(len(time)), 2):
        if not event[i]:
            continue
        if time[j] > time[i] or (time[j] == time[i] and not event[j]):
            pairs += 1
            if score[i] > score[j]:
                credit += 1
            elif score[i] == score[j]:
                credit += 0.5
    return pairs, credit


def brute_weighted(time, event, score, strata):
    num = den = 0.0
    n_strata = 0
    for key in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == key]
        n_strata += 1
        p, c = brute_pairs_and_credit(time[idx], event[idx], score[idx])
        num += c
        den += p
    return (num / den if den else None), int(den)


# --------------------------------------------------------------------------- data builders


def make_preds(n=60, landmarks=(0.0, 90.0), seed=1234, tied_times=True):
    rng = np.random.default_rng(seed)
    rows = []
    for lm in landmarks:
        for i in range(n):
            t = float(np.round(rng.exponential(300), -1) + 10) if tied_times else float(rng.exponential(300) + 10)
            rows.append({
                "patient_id": str(1000 + i), "landmark": lm, "log_risk_event": float(rng.normal()),
                "n_notes_used": 3, "model_id": "m", "time_from_index": t + lm,
                "time_from_landmark": t, "event": int(rng.random() < 0.6), "death_event": 0,
            })
    return pl.DataFrame(rows)


def make_cox(n=60, seed=7, folds=3, name="text", fold_seed=None):
    rng = np.random.default_rng(seed)
    frng = np.random.default_rng(seed if fold_seed is None else fold_seed)
    return pl.DataFrame({
        "DFCI_MRN": [1000 + i for i in range(n)],
        "outer_fold": frng.integers(0, folds, n).tolist(),
        f"{name}_risk_score": rng.normal(size=n).round(1).tolist(),  # rounded: tied risk scores
    })


def run_case(tmp_path, preds, text=None, base=None, text_test=None, base_test=None, report=None):
    risk = tmp_path / "risk"
    cox = tmp_path / "cox"
    risk.mkdir(exist_ok=True)
    cox.mkdir(exist_ok=True)
    pred_path = tmp_path / "test_predictions.csv"
    preds.write_csv(pred_path)
    if text is not None:
        text.write_csv(risk / "text_risk_scores.csv")
    if base is not None:
        base.write_csv(risk / "base_risk_scores.csv")
    if text_test is not None:
        text_test.write_csv(cox / "text_test.csv")
    if base_test is not None:
        base_test.write_csv(cox / "base_test.csv")
    return runner.build_comparison(str(pred_path), "treatment", str(tmp_path / "comparison.csv"),
                                   risk_dir=str(risk), cox_test_dir=str(cox), test_report=report)


def pick(out, model, landmark, scope="within_cox_folds"):
    sel = out.filter((pl.col("evaluation_scope") == scope) & (pl.col("model") == model)
                     & (pl.col("landmark") == landmark))
    assert sel.height == 1, (model, landmark, sel)
    return sel.row(0, named=True)


def lm_frame(preds, landmark):
    f = preds.filter(pl.col("landmark") == landmark).sort("patient_id")
    return f["time_from_landmark"].to_numpy(), f["event"].to_numpy(), f["log_risk_event"].to_numpy()


# --------------------------------------------------------------------------- pair counter


def test_admissible_pairs_matches_brute_force_and_lifelines():
    from lifelines.utils.concordance import _concordance_summary_statistics as lifelines_stats

    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(2, 30))
        t = rng.integers(1, 8, n).astype(float)  # heavy ties
        e = (rng.random(n) < 0.5).astype(int)
        s = rng.integers(0, 4, n).astype(float)
        pairs, _ = brute_pairs_and_credit(t, e, s)
        assert runner.admissible_pairs(t, e) == pairs
        if e.sum():
            assert runner.admissible_pairs(t, e) == int(lifelines_stats(t, -s, e)[2])


def test_admissible_pairs_edge_cases():
    assert runner.admissible_pairs([], []) == 0
    assert runner.admissible_pairs([5, 5, 5], [1, 1, 1]) == 0  # event/event ties excluded
    assert runner.admissible_pairs([5, 5], [1, 0]) == 1  # same-time event/censor included


# --------------------------------------------------------------------------- comparison


def test_pair_weighted_within_fold_c_index_matches_oracle(tmp_path):
    preds, text, base = make_preds(), make_cox(name="text"), make_cox(seed=8, name="base", fold_seed=7)
    out = run_case(tmp_path, preds, text, base)
    for lm in (0.0, 90.0):
        t, e, s = lm_frame(preds, lm)
        folds = text.sort("DFCI_MRN")["outer_fold"].to_numpy()  # same folds for both (same fold_seed)
        scores = {"transformer": s, "cox_text_oof": text.sort("DFCI_MRN")["text_risk_score"].to_numpy(),
                  "cox_base_oof": base.sort("DFCI_MRN")["base_risk_score"].to_numpy()}
        pair_counts = set()
        for model, score in scores.items():
            expected_c, expected_pairs = brute_weighted(t, e, score, folds.tolist())
            row = pick(out, model, lm)
            assert row["n_comparable_pairs"] == expected_pairs
            assert row["c_index"] == pytest.approx(expected_c, abs=1e-12)
            assert row["n_patients"] == len(t) and row["n_strata"] == 3
            pair_counts.add(row["n_comparable_pairs"])
        assert len(pair_counts) == 1  # one common pair set for every model


def test_constant_shift_of_one_cox_fold_leaves_comparison_unchanged(tmp_path):
    preds, text = make_preds(), make_cox(name="text")
    (tmp_path / "a").mkdir()
    base_out = run_case(tmp_path / "a", preds, text)
    shifted = text.with_columns(
        pl.when(pl.col("outer_fold") == 1).then(pl.col("text_risk_score") + 100.0)
        .otherwise(pl.col("text_risk_score")).alias("text_risk_score")
    )
    (tmp_path / "b").mkdir()
    shifted_out = run_case(tmp_path / "b", preds, shifted)
    for lm in (0.0, 90.0):
        assert pick(shifted_out, "cox_text_oof", lm)["c_index"] == pytest.approx(
            pick(base_out, "cox_text_oof", lm)["c_index"], abs=1e-12)
    # A pooled raw-score C-index would have changed under that shift.
    t, e, _ = lm_frame(preds, 0.0)
    raw = text.sort("DFCI_MRN")["text_risk_score"].to_numpy()
    shifted_raw = shifted.sort("DFCI_MRN")["text_risk_score"].to_numpy()
    assert safe_c_index(t, raw, e)[0] != pytest.approx(safe_c_index(t, shifted_raw, e)[0])


def test_differing_text_and_base_folds_use_joint_strata(tmp_path):
    preds = make_preds()
    text = make_cox(name="text")
    base = make_cox(seed=8, name="base", fold_seed=99)
    out = run_case(tmp_path, preds, text, base)
    t, e, s = lm_frame(preds, 0.0)
    joint = [f"{a}|{b}" for a, b in zip(text.sort("DFCI_MRN")["outer_fold"].to_list(),
                                        base.sort("DFCI_MRN")["outer_fold"].to_list())]
    expected_c, expected_pairs = brute_weighted(t, e, s, joint)
    row = pick(out, "transformer", 0.0)
    assert row["strata_definition"] == "text_fold|base_fold"
    assert row["n_strata"] == len(set(joint))
    assert row["n_comparable_pairs"] == expected_pairs
    assert row["c_index"] == pytest.approx(expected_c, abs=1e-12)


def test_single_stratum_equals_plain_safe_c_index(tmp_path):
    preds = make_preds(landmarks=(0.0,))
    text = make_cox(folds=1)  # every patient in fold 0
    out = run_case(tmp_path, preds, text)
    t, e, s = lm_frame(preds, 0.0)
    row = pick(out, "transformer", 0.0)
    assert row["n_strata"] == 1
    assert row["c_index"] == pytest.approx(safe_c_index(t, s, e)[0], abs=1e-12)


def test_uninformative_fold_contributes_half_credit_with_its_weight(tmp_path):
    preds = make_preds(landmarks=(0.0,), tied_times=False)
    text = make_cox().with_columns(
        pl.when(pl.col("outer_fold") == 2).then(0.0).otherwise(pl.col("text_risk_score")).alias("text_risk_score")
    )
    out = run_case(tmp_path, preds, text)
    t, e, _ = lm_frame(preds, 0.0)
    score = text.sort("DFCI_MRN")["text_risk_score"].to_numpy()
    folds = text.sort("DFCI_MRN")["outer_fold"].to_list()
    expected_c, expected_pairs = brute_weighted(t, e, score, folds)
    row = pick(out, "cox_text_oof", 0.0)
    assert row["c_index"] == pytest.approx(expected_c, abs=1e-12)
    assert row["n_comparable_pairs"] == expected_pairs


def test_nonfinite_cox_scores_excluded_jointly_with_reconciled_counts(tmp_path):
    preds = make_preds(landmarks=(0.0,))
    text = make_cox(name="text")
    scores = text["text_risk_score"].to_list()
    scores[0] = float("nan")
    scores[1] = float("inf")
    text = text.with_columns(pl.Series("text_risk_score", scores))
    text = text.filter(pl.col("DFCI_MRN") != 1002)  # missing row
    text = text.with_columns(pl.when(pl.col("DFCI_MRN") == 1003).then(-1).otherwise(pl.col("outer_fold")).alias("outer_fold"))
    out = run_case(tmp_path, preds, text, make_cox(seed=8, name="base"))
    tr, ct, cb = (pick(out, m, 0.0) for m in ("transformer", "cox_text_oof", "cox_base_oof"))
    assert tr["n_patients"] == ct["n_patients"] == cb["n_patients"] == 60 - 4
    assert (tr["excluded_missing_score"], tr["excluded_nonfinite_score"], tr["excluded_invalid_fold"]) == (1, 2, 1)
    assert tr["n_all_test"] == tr["n_patients"] + 1 + 2 + 1
    assert tr["n_comparable_pairs"] == ct["n_comparable_pairs"] == cb["n_comparable_pairs"]


def test_missing_optional_cox_file_does_not_shrink_the_other_comparator(tmp_path):
    preds, text = make_preds(), make_cox(name="text")
    out = run_case(tmp_path, preds, text)  # no base file
    base = pick(out, "cox_base_oof", 0.0)
    assert base["c_index"] is None and base["c_index_reason"] == "missing_cox_file"
    tr = pick(out, "transformer", 0.0)
    assert tr["n_patients"] == 60 and tr["comparators_available"] == "cox_text_oof"
    assert tr["strata_definition"] == "text_fold"
    assert pick(out, "cox_text_oof", 0.0)["c_index"] is not None


def test_no_cox_files_keeps_all_test_transformer_metric(tmp_path):
    preds = make_preds()
    out = run_case(tmp_path, preds)
    for lm in (0.0, 90.0):
        row = pick(out, "transformer", lm)
        assert row["c_index"] is None and row["c_index_reason"] == "no_cox_comparator_available"
        t, e, s = lm_frame(preds, lm)
        all_test = pick(out, "transformer", lm, scope="all_test")
        assert all_test["c_index"] == pytest.approx(safe_c_index(t, s, e)[0])
        assert all_test["n_patients"] == 60
    assert out.filter(pl.col("evaluation_scope") == "reported_cox_holdout")["c_index_reason"].to_list() == [
        "missing_cox_file", "missing_cox_file"]


def test_duplicate_cox_ids_make_the_comparator_invalid(tmp_path):
    preds = make_preds(landmarks=(0.0,))
    text = pl.concat([make_cox(name="text"), make_cox(name="text").head(1)])
    out = run_case(tmp_path, preds, text, make_cox(seed=8, name="base"))
    assert pick(out, "cox_text_oof", 0.0)["c_index_reason"] == "invalid_cox_file"
    assert pick(out, "transformer", 0.0)["comparators_available"] == "cox_base_oof"


def test_missing_fold_column_is_invalid(tmp_path):
    preds = make_preds(landmarks=(0.0,))
    out = run_case(tmp_path, preds, make_cox(name="text").drop("outer_fold"))
    assert pick(out, "cox_text_oof", 0.0)["c_index_reason"] == "invalid_cox_file"


def test_empty_intersection_yields_null_reasons_not_failure(tmp_path):
    preds = make_preds(landmarks=(0.0,))
    text = make_cox(name="text").with_columns((pl.col("DFCI_MRN") + 5000).alias("DFCI_MRN"))
    out = run_case(tmp_path, preds, text)
    for model in ("transformer", "cox_text_oof"):
        row = pick(out, model, 0.0)
        assert row["c_index"] is None and row["c_index_reason"] == "empty_intersection"
        assert row["n_patients"] == 0 and row["excluded_missing_score"] == 60
    assert pick(out, "transformer", 0.0, scope="all_test")["c_index"] is not None


def test_no_events_and_no_comparable_pairs_reasons(tmp_path):
    preds = make_preds(landmarks=(0.0, 90.0)).with_columns(
        pl.when(pl.col("landmark") == 0.0).then(0).otherwise(1).alias("event"),
        pl.when(pl.col("landmark") == 0.0).then(pl.col("time_from_landmark"))
        .otherwise(pl.lit(50.0)).alias("time_from_landmark"),  # everyone an event at the same time
    )
    out = run_case(tmp_path, preds, make_cox(name="text"))
    assert pick(out, "transformer", 0.0)["c_index_reason"] == "no_primary_events"
    row = pick(out, "cox_text_oof", 90.0)
    assert row["c_index"] is None and row["c_index_reason"] == "no_comparable_pairs"
    assert row["n_comparable_pairs"] == 0


@pytest.mark.parametrize("empty_predictions", [False, True])
@pytest.mark.parametrize("with_cox", [False, True])
def test_landmarks_without_eligible_patients_retain_comparison_rows(tmp_path, empty_predictions, with_cox):
    preds = make_preds(landmarks=(0.0,))
    if empty_predictions:
        preds = preds.head(0)
    time, event, score = lm_frame(preds, 0.0)
    c, reason = safe_c_index(time, score, event)
    c = float(c) if np.isfinite(c) else None
    report = {"weighted_test_c_index": c, "landmark_metrics": [
        {"landmark": 0.0, "c_index": c, "c_index_reason": reason,
         "n_eligible": preds.height, "n_events": int(preds["event"].sum())},
        {"landmark": 365.0, "c_index": None, "c_index_reason": "fewer_than_two_patients",
         "n_eligible": 0, "n_events": 0},
    ]}
    out = run_case(tmp_path, preds,
                   text=make_cox(name="text") if with_cox else None,
                   base=make_cox(name="base") if with_cox else None, report=report)
    empty_landmarks = (0.0, 365.0) if empty_predictions else (365.0,)
    for landmark in empty_landmarks:
        for model in ("transformer", "cox_text_oof", "cox_base_oof"):
            row = pick(out, model, landmark)
            assert row["c_index"] is None
            assert row["n_all_test"] == 0
            if with_cox:
                assert row["c_index_reason"] == "empty_intersection"
                assert row["n_patients"] == row["n_events"] == row["n_comparable_pairs"] == 0
            else:
                assert row["c_index_reason"] == (
                    "no_cox_comparator_available" if model == "transformer" else "missing_cox_file"
                )
        assert pick(out, "transformer", landmark, scope="all_test")["n_patients"] == 0
    assert out.filter(pl.col("evaluation_scope") == "within_cox_folds").height == 6


def test_all_test_and_reported_holdout_scopes_are_separate(tmp_path):
    preds = make_preds()
    report = {"weighted_test_c_index": 0.61, "model_id": "m", "landmark_metrics": [
        {"landmark": 0.0, "c_index": 0.6, "c_index_reason": None, "n_eligible": 60, "n_events": 30},
        {"landmark": 90.0, "c_index": None, "c_index_reason": "no_primary_events", "n_eligible": 10, "n_events": 0},
    ]}
    out = run_case(tmp_path, preds, make_cox(name="text"),
                   text_test=pl.DataFrame({"mean_c_index": [0.7]}), base_test=pl.DataFrame({"mean_c_index": [0.65]}),
                   report=report)
    assert pick(out, "transformer", 0.0, scope="all_test")["c_index"] == 0.6
    assert pick(out, "transformer", 90.0, scope="all_test")["c_index_reason"] == "no_primary_events"
    weighted = out.filter(pl.col("model") == "transformer_weighted_by_eligible").row(0, named=True)
    assert weighted["c_index"] == 0.61
    reported = out.filter(pl.col("evaluation_scope") == "reported_cox_holdout")
    assert dict(zip(reported["model"], reported["c_index"])) == {"cox_text_test": 0.7, "cox_base_test": 0.65}
    assert set(out["evaluation_scope"].unique()) == {"all_test", "within_cox_folds", "reported_cox_holdout"}
    assert (tmp_path / "comparison.csv").exists()


# --------------------------------------------------------------------------- checkpoint selection


@pytest.fixture
def dirs(tmp_path):
    d = {k: str(tmp_path / k) for k in ("root", "search", "single_run", "test")}
    for k in ("search", "single_run"):
        os.makedirs(d[k])
    return d


def touch(path):
    open(path, "wb").close()
    return path


def test_pick_checkpoint_neither_one_both_and_explicit(dirs, tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        runner.pick_checkpoint(None, dirs)
    only_search = touch(os.path.join(dirs["search"], "best_model.pt"))
    assert runner.pick_checkpoint(None, dirs) == only_search
    new_single = touch(os.path.join(dirs["single_run"], "best_model.pt"))
    with pytest.raises(ValueError, match="pass --checkpoint") as err:  # stale search vs new single run
        runner.pick_checkpoint(None, dirs)
    assert only_search in str(err.value) and new_single in str(err.value)
    explicit = touch(str(tmp_path / "other.pt"))
    assert runner.pick_checkpoint(explicit, dirs) == explicit
    with pytest.raises(FileNotFoundError):
        runner.pick_checkpoint(str(tmp_path / "nope.pt"), dirs)
    os.remove(only_search)
    assert runner.pick_checkpoint(None, dirs) == new_single


@pytest.fixture
def fake_pipeline(monkeypatch, dirs, tmp_path):
    calls = {}
    monkeypatch.setattr(runner, "run_dirs", lambda anchor, tag: dirs)
    monkeypatch.setattr(runner, "load_inputs", lambda d, anchor: (
        object(), {"signature": {"anchor": anchor}, "split_check": "x", "counts": {"patients": 1}}
    ))

    def fake_search(data, config, space, n, seed, out, resume=False):
        path = touch(os.path.join(out, "best_model.pt"))
        calls["search_out"] = out
        return {"checkpoint_path": path, "selected_trial": 0, "validation_score": 0.6}

    def fake_test(checkpoint, data, out, device=None, num_threads=None):
        calls["test_checkpoint"] = checkpoint
        calls["test_num_threads"] = num_threads
        return {"model_id": "m", "weighted_test_c_index": 0.6, "landmark_metrics": []}

    monkeypatch.setattr(runner, "run_search", fake_search)
    monkeypatch.setattr(runner, "final_test_evaluation", fake_test)
    monkeypatch.setattr(runner, "build_comparison", lambda *a, **k: pl.DataFrame({"x": [1]}))
    return calls


def test_all_stage_tests_its_own_search_checkpoint_even_with_another_present(fake_pipeline, dirs):
    stale = touch(os.path.join(dirs["single_run"], "best_model.pt"))
    runner.main(["--stage", "all", "--device", "cpu", "--num-trials", "1", "--num-threads", "2"])
    assert fake_pipeline["test_checkpoint"] == os.path.join(dirs["search"], "best_model.pt")
    assert fake_pipeline["test_checkpoint"] != stale
    assert fake_pipeline["test_num_threads"] == 2


def test_test_stage_uses_explicit_checkpoint_and_refuses_ambiguity(fake_pipeline, dirs, tmp_path):
    touch(os.path.join(dirs["search"], "best_model.pt"))
    chosen = touch(os.path.join(dirs["single_run"], "best_model.pt"))
    with pytest.raises(ValueError, match="pass --checkpoint"):
        runner.main(["--stage", "test", "--device", "cpu"])
    assert "test_checkpoint" not in fake_pipeline
    runner.main(["--stage", "test", "--device", "cpu", "--checkpoint", chosen, "--num-threads", "1"])
    assert fake_pipeline["test_checkpoint"] == chosen
    assert fake_pipeline["test_num_threads"] == 1


def test_test_stage_uses_slurm_thread_budget(fake_pipeline, dirs, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    touch(os.path.join(dirs["search"], "best_model.pt"))
    runner.main(["--stage", "test", "--device", "cpu"])
    assert fake_pipeline["test_num_threads"] == 2


@pytest.mark.parametrize("stage", ["search", "train", "test", "all"])
@pytest.mark.parametrize("prepared_anchor,requested_anchor", [
    ("treatment", "sequencing"), ("sequencing", "treatment"), (None, "treatment"),
])
def test_anchor_mismatch_fails_before_data_loading_or_output_changes(
    tmp_path, monkeypatch, stage, prepared_anchor, requested_anchor
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = {"signature": {"schema_version": runner.SCHEMA_VERSION, "anchor": prepared_anchor}}
    (inputs / runner.MANIFEST_NAME).write_text(json.dumps(manifest))
    outputs = {k: str(tmp_path / k) for k in ("root", "search", "single_run", "test")}
    for path in outputs.values():
        os.makedirs(path)
        with open(os.path.join(path, "sentinel"), "w") as handle:
            handle.write("existing results")
    monkeypatch.setattr(runner, "run_dirs", lambda anchor, tag: outputs)
    monkeypatch.setattr(runner, "load_data", lambda *a, **k: pytest.fail("loaded mismatched inputs"))
    with pytest.raises(ValueError, match="prepared for anchor"):
        runner.main(["--stage", stage, "--anchor", requested_anchor, "--inputs-dir", str(inputs),
                     "--device", "cpu", "--num-threads", "1", "--overwrite"])
    for path in outputs.values():
        assert os.listdir(path) == ["sentinel"]
        with open(os.path.join(path, "sentinel")) as handle:
            assert handle.read() == "existing results"


def test_checkpoint_flag_rejected_outside_test_stage():
    with pytest.raises(SystemExit):
        runner.main(["--stage", "all", "--checkpoint", "x.pt"])


def test_prepare_output_dir_guards_existing_output(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "f").write_text("x")
    with pytest.raises(FileExistsError):
        runner.prepare_output_dir(str(out), overwrite=False, resume=False)
    assert runner.prepare_output_dir(str(out), overwrite=False, resume=True) == str(out)
    runner.prepare_output_dir(str(out), overwrite=True, resume=False)
    assert os.listdir(out) == []


def test_allow_nondeterministic_wraps_and_restores():
    import torch

    original = torch.use_deterministic_algorithms
    torch.use_deterministic_algorithms(False)
    with runner.nondeterministic_ok(True):
        torch.use_deterministic_algorithms(True)  # what seed_everything does
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.use_deterministic_algorithms is original
    assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
