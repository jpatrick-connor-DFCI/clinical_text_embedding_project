from figures.prep import parallel
from figures.prep.parallel import fit_thread_limit


def test_fit_thread_limit_caps_workers_to_nproc_headroom():
    # 900-thread limit with 500 in use: (900 - 500 - 64) // (4 + 6) = 33 -> keep 16.
    assert fit_thread_limit(16, 1, 900, 500) == 16
    # 6 threads per worker: (900 - 500 - 64) // 40 = 8.
    assert fit_thread_limit(16, 6, 900, 500) == 8
    # No head-room still leaves one (serial) worker.
    assert fit_thread_limit(16, 1, 900, 890) == 1


def test_fit_thread_limit_ignores_unknown_or_unlimited():
    assert fit_thread_limit(16, 6, None, 500) == 16
    assert fit_thread_limit(16, 6, 900, None) == 16
    assert fit_thread_limit(1, 6, 900, 890) == 1


def test_worker_threads_default_and_override(monkeypatch):
    monkeypatch.delenv(parallel.WORKER_THREADS_ENV, raising=False)
    assert parallel.worker_threads() == 1
    monkeypatch.setenv(parallel.WORKER_THREADS_ENV, "4")
    assert parallel.worker_threads() == 4
    monkeypatch.setenv(parallel.WORKER_THREADS_ENV, "x")
    assert parallel.worker_threads() == 1


def test_process_pool_runs_serially_when_limit_is_exhausted(monkeypatch, capsys):
    monkeypatch.setattr(parallel, "_nproc_limit", lambda: 900)
    monkeypatch.setattr(parallel, "_user_thread_count", lambda: 880)
    with parallel.process_pool(16) as pool:
        assert pool is None
    assert "limiting to 1 worker(s): 880 of the 900 threads" in capsys.readouterr().out
