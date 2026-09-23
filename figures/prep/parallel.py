"""Process pools for the CPU-bound figure-prep loops (C-index evaluation, Cox fits).

sksurv's concordance and lifelines' Newton steps hold the GIL for most of their
runtime, so threads barely overlap them; these helpers use processes instead.
Workers are started with "spawn" (forking a process that has initialized
polars' thread pool can deadlock). Each worker's polars/BLAS pools default to one
thread (FIGURE_PREP_WORKER_THREADS overrides): the per-task work is
single-threaded, and every extra thread slot costs a worker about five OS
threads (polars compute + async executors, OpenBLAS, OpenMP). Results always come
back in submission order, so parallel output is identical to the serial loop.

The worker count comes from FIGURE_PREP_N_JOBS; otherwise it is
DEFAULT_WORKERS (16), capped at the CPU allocation (SLURM_CPUS_PER_TASK, else the
machine's core count). On Linux it is further capped to fit the per-user process
limit (RLIMIT_NPROC, which counts threads; 900 on some cluster nodes) given the
threads the user is already running, since exceeding it makes thread creation
fail inside polars/OpenBLAS. With one worker nothing is spawned and callers run
their serial path.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import resource
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

N_JOBS_ENV = "FIGURE_PREP_N_JOBS"
WORKER_THREADS_ENV = "FIGURE_PREP_WORKER_THREADS"
# Spawned workers each import polars/sksurv/lifelines, so an uncapped default on
# a large node spends minutes (and many GB) just starting processes.
DEFAULT_WORKERS = 16
# Measured OS threads per spawned worker: about 2 + 5 per thread slot (7 at 1,
# 33 at 6). The estimate rounds up and adds the parent's executor threads.
_THREADS_PER_WORKER_BASE = 4
_THREADS_PER_SLOT = 6
# Head-room left under RLIMIT_NPROC for the notebook, shells and anything else
# the user starts while the pool runs.
_NPROC_RESERVE = 64
_THREAD_VARS = (
    "POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _allocated_cpus() -> int:
    raw = os.getenv("SLURM_CPUS_PER_TASK")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return os.cpu_count() or 1


def _positive_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring", name, raw)
        return None
    return value if value > 0 else None


def worker_threads() -> int:
    """polars/BLAS threads per worker: FIGURE_PREP_WORKER_THREADS, else 1."""
    return _positive_int_env(WORKER_THREADS_ENV) or 1


def _user_thread_count() -> int | None:
    """OS threads currently owned by this user (Linux /proc), else None."""
    if not os.path.isdir("/proc/self/task"):
        return None
    uid, total = os.getuid(), 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/status") as fh:
                fields = dict(line.split(":", 1) for line in fh if ":" in line)
            if int(fields["Uid"].split()[0]) == uid:
                total += int(fields["Threads"])
        except (OSError, KeyError, ValueError):
            continue
    return total


def fit_thread_limit(n_workers: int, threads: int, limit: int | None, in_use: int | None) -> int:
    """Largest worker count <= n_workers whose threads fit under `limit`."""
    if n_workers <= 1 or limit is None or in_use is None:
        return n_workers
    per_worker = _THREADS_PER_WORKER_BASE + _THREADS_PER_SLOT * threads
    return max(1, min(n_workers, (limit - in_use - _NPROC_RESERVE) // per_worker))


def _nproc_limit() -> int | None:
    soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    return None if soft == resource.RLIM_INFINITY else soft


def resolve_workers(n_jobs: int | None = None) -> int:
    """Worker count: explicit n_jobs, else FIGURE_PREP_N_JOBS, else
    DEFAULT_WORKERS capped at the CPU allocation."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    return _positive_int_env(N_JOBS_ENV) or min(DEFAULT_WORKERS, _allocated_cpus())


@contextmanager
def process_pool(n_workers: int, *, initializer=None, initargs: tuple = ()) -> Iterator[Executor | None]:
    """Yield a spawn-based ProcessPoolExecutor, or None when n_workers <= 1.

    n_workers is reduced, with a notice, when it would not fit the per-user
    thread limit. Spawned children inherit os.environ as it is when each
    starts, so the per-worker thread caps are set for the pool's lifetime and
    restored after.
    """
    threads = worker_threads()
    if n_workers > 1:
        limit, in_use = _nproc_limit(), _user_thread_count()
        fitted = fit_thread_limit(n_workers, threads, limit, in_use)
        if fitted < n_workers:
            print(f"  limiting to {fitted} worker(s): {in_use} of the {limit} threads allowed "
                  f"per user are already in use (ulimit -u)", flush=True)
            n_workers = fitted
    if n_workers <= 1:
        yield None
        return
    print(f"  starting {n_workers} worker process(es), {threads} thread(s) each", flush=True)
    saved = {var: os.environ.get(var) for var in _THREAD_VARS}
    os.environ.update({var: str(threads) for var in _THREAD_VARS})
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
