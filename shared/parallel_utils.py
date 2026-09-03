"""Worker-pool sizing and oversubscription control for the stratum-parallel model scripts.

Two things every fan-out in this project has had to get right, factored out of the copies that
were drifting in ``generate_mortality_trajectories`` and the within-vs-pan scripts:

1. **Sizing.** Leave a core free and budget RAM per worker, honouring an explicit env override.
2. **Oversubscription.** A spawned child builds its polars/Rayon and BLAS thread pools while
   importing those modules -- before any ``initializer`` argument can run. Limiting threads
   inside the worker is therefore always too late. The variables must be in the environment
   the child inherits, which means setting them in the parent before the pool is created.

Getting (2) wrong is not a slowdown, it is a crash: N workers each sizing a 64-thread Rayon pool
plus a 64-thread BLAS pool exhausts ``RLIMIT_NPROC`` on a fat node, and the failure surfaces as a
polars ``ThreadPoolBuildError`` followed by a poisoned ``LazyLock`` and an unimportable numpy.
"""

from __future__ import annotations

import os

# Every knob a worker might use to size a thread pool to the whole machine.
SINGLE_THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def resolve_workers(n_tasks: int, *, env_var: str, worker_mem_gb: float = 8.0) -> int:
    """Size a worker pool: explicit override, else one core free and ``worker_mem_gb`` each.

    Never returns more workers than there are tasks, and never fewer than 1. An unparseable
    override is ignored rather than raising -- a bad env var should not kill a multi-hour run
    before it starts.
    """
    if n_tasks <= 0:
        return 1
    override = os.environ.get(env_var, "").strip()
    if override:
        try:
            return max(1, min(int(override), n_tasks))
        except ValueError:
            pass
    workers = max(1, (os.cpu_count() or 1) - 1)
    try:
        # Linux-only; on a cgroup-limited node this tracks the real allowance better than total
        # system memory. Skipped silently where unavailable (e.g. macOS).
        available_gb = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')) / 1024 ** 3
        workers = max(1, min(workers, int(available_gb // worker_mem_gb)))
    except (ValueError, OSError, AttributeError):
        pass
    return max(1, min(workers, n_tasks))


def set_single_thread_env() -> None:
    """Pin every BLAS/Rayon knob to one thread in this process's environment.

    Call in the PARENT immediately before creating a spawn-based pool: children inherit
    ``os.environ``, and by the time a worker's initializer runs its thread pools are already
    built. Assignment, not ``setdefault`` -- an inherited wide value is exactly what this
    exists to override.

    The parent's own already-built pools keep their threads, so calling this after the parent's
    heavy single-process work is finished costs it nothing.
    """
    for var in SINGLE_THREAD_VARS:
        os.environ[var] = "1"
