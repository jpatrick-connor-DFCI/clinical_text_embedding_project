"""Regression tests for resumable within-vs-pan stage checkpoints."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest


CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "survival" / "checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_under_test", CHECKPOINT_PATH)
checkpoint_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checkpoint_module)
RunCheckpoint = checkpoint_module.RunCheckpoint


def _scores(score_name):
    train = pl.DataFrame({"DFCI_MRN": [1, 2], score_name: [0.1, 0.2]})
    held = pl.DataFrame({"DFCI_MRN": [3], score_name: [0.3]})
    return train, held


def test_restart_can_load_within_stage_when_matched_pan_is_missing(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    fingerprint = {"script": "test", "seed": 1234}
    within_train, within_held = _scores("within_risk_score")

    first_run = RunCheckpoint(checkpoint_dir, fingerprint)
    first_run.save_stratum("lung", within_train, within_held)

    # Simulate interruption before the matched-pan stage was saved.
    resumed_run = RunCheckpoint(checkpoint_dir, fingerprint)
    assert resumed_run.stratum_done("lung")
    assert not resumed_run.stratum_done("lung__matched_pan__")
    loaded_train, loaded_held = resumed_run.load_stratum("lung")

    assert loaded_train.equals(within_train)
    assert loaded_held.equals(within_held)
    assert resumed_run.counts() == (1, 0, 0)


def test_restart_loads_both_independent_stages_and_counts_only_within(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    fingerprint = {"script": "test", "seed": 1234}
    within_train, within_held = _scores("within_risk_score")
    pan_train, pan_held = _scores("pan_risk_score")

    first_run = RunCheckpoint(checkpoint_dir, fingerprint)
    first_run.save_stratum("lung", within_train, within_held)
    first_run.save_stratum("lung__matched_pan__", pan_train, pan_held)
    first_run.update_stratum_meta("lung", {"c_pan": 0.6, "c_within": 0.7})

    resumed_run = RunCheckpoint(checkpoint_dir, fingerprint)
    resumed_run.load_stratum("lung")
    resumed_run.load_stratum("lung__matched_pan__")

    assert resumed_run.counts() == (1, 0, 0)
    assert resumed_run.metadata("lung")["delta_c"] == pytest.approx(0.1)


def test_done_manifest_without_both_score_files_is_not_resumable(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = RunCheckpoint(checkpoint_dir, {"script": "test"})
    train, held = _scores("risk_score")
    checkpoint.save_stratum("lung", train, held)

    Path(checkpoint._held_path("lung")).unlink()

    resumed_run = RunCheckpoint(checkpoint_dir, {"script": "test"})
    assert resumed_run.status("lung") == "done"
    assert not resumed_run.stratum_done("lung")


def test_changed_fingerprint_does_not_reuse_stale_stage(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    train, held = _scores("risk_score")
    RunCheckpoint(checkpoint_dir, {"data_hash": "old"}).save_stratum("lung", train, held)

    changed_run = RunCheckpoint(checkpoint_dir, {"data_hash": "new"})

    assert changed_run.status("lung") is None
    assert not changed_run.stratum_done("lung")
