"""Exercise the shell renderer without requiring an R installation or figure data."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("selected", [[], ["plot_figure_1.R", "plot_figure_2.R"]])
def test_render_once_with_cindex_despite_inherited_auc(tmp_path, selected):
    repo = Path(__file__).resolve().parents[1]
    log = tmp_path / "calls.jsonl"
    rscript = tmp_path / "Rscript"
    rscript.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CTEP_TEST_RENDER_LOG'], 'a') as output:\n"
        "    output.write(json.dumps([sys.argv[1], os.environ['MANUSCRIPT_METRIC'], "
        "os.environ['MANUSCRIPT_METRICS']]) + '\\n')\n"
    )
    rscript.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        MANUSCRIPT_METRIC="auc",
        MANUSCRIPT_METRICS="auc cindex",
        CTEP_TEST_RENDER_LOG=str(log),
    )

    result = subprocess.run(
        ["bash", str(repo / "R/render_all_figures.sh"), *selected],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    expected = selected or sorted(p.name for p in (repo / "R").glob("plot_figure_*.R"))
    assert calls == [[f"R/{script}", "cindex", "cindex"] for script in expected]
    assert str(repo / "R/render_all_figures.sh") in result.stdout
