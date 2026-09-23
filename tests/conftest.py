import pytest


@pytest.fixture(autouse=True)
def _serial_figure_prep(monkeypatch):
    """Run figure prep serially: spawned workers would not see tests' monkeypatches.

    Tests that exercise the process pool pass n_jobs explicitly.
    """
    monkeypatch.setenv("FIGURE_PREP_N_JOBS", "1")
