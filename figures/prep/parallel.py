"""Process pools for the CPU-bound figure-prep loops (C-index evaluation, Cox fits).

sksurv's concordance and lifelines' Newton steps hold the GIL for most of their
runtime, so threads barely overlap them; these helpers use processes instead.
Workers are started with "spawn" (forking a process that has initialized
polars' thread pool can deadlock), and each worker's polars/BLAS thread pools
are sized to its share of the allocation so N workers do not oversubscribe the
node. Results always come back in submission order, so parallel output is
identical to the serial loop.

The worker count comes from FIGURE_PREP_N_JOBS, then SLURM_CPUS_PER_TASK, then
the machine's core count. With one worker nothing is spawned and callers run
their serial path.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

N_JOBS_ENV = "FIGURE_PREP_N_JOBS"
_THREAD_VARS = (
    "POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _allocated_cpus() -> int:
    raw = os.getenv("SLURM_CPUS_PER_TASK")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return os.cpu_count() or 1


def resolve_workers(n_jobs: int | None = None) -> int:
    """Worker count: explicit n_jobs, else FIGURE_PREP_N_JOBS, else the CPU allocation."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    raw = os.getenv(N_JOBS_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("%s=%r is not an integer; ignoring", N_JOBS_ENV, raw)
        else:
            if value > 0:
                return value
    return _allocated_cpus()


@contextmanager
def process_pool(n_workers: int, *, initializer=None, initargs: tuple = ()) -> Iterator[Executor | None]:
    """Yield a spawn-based ProcessPoolExecutor, or None when n_workers <= 1.

    Spawned children inherit os.environ as it is when each starts, so the
    per-worker thread caps are set for the pool's lifetime and restored after.
    """
    if n_workers <= 1:
        yield None
        return
    threads = str(max(1, _allocated_cpus() // n_workers))
    saved = {var: os.environ.get(var) for var in _THREAD_VARS}
    os.environ.update({var: threads for var in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initializer, initargs=initargs) as pool:
            yield pool
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
