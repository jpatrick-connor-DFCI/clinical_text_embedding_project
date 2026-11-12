"""Execute the saved note-transformer summary without training or running other cells."""

import json
import os
from pathlib import Path

import polars as pl
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/2_models/07_note_transformer_os.ipynb"


@pytest.mark.parametrize("weighted_score, expected_label", [(None, "unavailable"), (0.75, "0.7500")])
def test_summary_displays_score_and_continues_diagnostics(tmp_path, capsys, weighted_score, expected_label):
    notebook = json.loads(NOTEBOOK.read_text())
    summary_heading = next(
        i for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## Summary\n")
    )
    summary = next(
        cell for cell in notebook["cells"][summary_heading + 1:] if cell["cell_type"] == "code"
    )
    dirs = {name: str(tmp_path / name) for name in ("search", "single_run", "test")}
    test_dir = Path(dirs["test"])
    test_dir.mkdir()
    reason = "no_primary_events" if weighted_score is None else None
    report = {
        "model_id": "selected-model",
        "checkpoint_path": str(tmp_path / "best_model.pt"),
        "weighted_test_c_index": weighted_score,
        "landmark_metrics": [{
            "landmark": 365.0,
            "n_eligible": 3,
            "n_events": 0 if weighted_score is None else 2,
            "c_index": weighted_score,
            "c_index_reason": reason,
        }],
    }
    (test_dir / "test_metrics.json").write_text(json.dumps(report))
    pl.DataFrame({
        "evaluation_scope": ["all_test", "within_cox_folds"],
        "model": ["transformer", "transformer"],
        "landmark": [365.0, 365.0],
        "c_index": [weighted_score, weighted_score],
        "c_index_reason": [reason, reason],
    }).write_csv(test_dir / "comparison.csv")
    reported_outputs = []
    namespace = {
        "os": os,
        "json": json,
        "pl": pl,
        "DIRS": dirs,
        "INPUTS_DIR": str(tmp_path / "inputs"),
        "MANIFEST_NAME": "prepare_manifest.json",
        "report_outputs": reported_outputs.extend,
    }

    with pl.Config():
        exec(compile("".join(summary["source"]), str(NOTEBOOK), "exec"), namespace)

    output = capsys.readouterr().out
    assert f"weighted test C-index: {expected_label}" in output
    assert "selected-model" in output
    assert "Evaluation scope: all_test" in output
    assert "Evaluation scope: within_cox_folds" in output
    if reason:
        assert reason in output
    assert ("test metrics", str(test_dir / "test_metrics.json")) in reported_outputs
    assert ("comparison", str(test_dir / "comparison.csv")) in reported_outputs
