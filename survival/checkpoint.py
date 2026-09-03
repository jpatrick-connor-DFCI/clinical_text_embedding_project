"""Resumable checkpointing for the long-running trajectory model scripts.

Holds two stores. ``RunCheckpoint`` serves the within-vs-pan cancer/treatment scripts, which
produce a train/held-out score *pair* per stratum. ``LandmarkCheckpoint`` serves
``generate_mortality_trajectories``, which produces a single score column per landmark; both
share the fingerprint-staleness rule described below.

The within-vs-pan cancer/treatment scripts run for hours: one elastic-net grid search per
stratum plus the pan model. ``RunCheckpoint`` writes each expensive stage's results to disk as
it completes (pan scores, per-stratum train/held-out risk scores, a progress log, and a metric
manifest) so a crashed or interrupted run can resume — completed strata are reloaded from CSV
and skipped instead of refit.

Only CSVs are persisted, never fitted model objects: the held-out predictions are written at the
same time as the train scores (while the model is still in memory), so nothing downstream needs
the model back. Resume is sound because the upstream pipeline is deterministic (fixed-seed
train/held split, deterministic scaling/imputation), so reloaded scores stay valid across runs.

A ``fingerprint`` dict (cohort sizes, feature counts, stratum list, seed) is stored on first run;
if a later run's fingerprint differs, the stale checkpoints are ignored and the run starts fresh.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime

import polars as pl

PAN_KEY = '__pan__'
LANDMARK_MANIFEST_COLS = ['month', 'status', 'reason', 'n_at_risk', 'n_scored', 'elapsed_s', 'ts']
MANIFEST_COLS = ['key', 'status', 'reason', 'n_train', 'n_held',
                 'l1', 'alpha', 'c_pan', 'c_within', 'delta_c', 'elapsed_s', 'ts']


def dataframe_fingerprint(df, columns):
    """Content hash for checkpoint inputs, including column order and row index."""
    selected = list(dict.fromkeys(columns))
    missing = [column for column in selected if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot fingerprint missing columns: {missing}")
    # NOTE: previously used pd.util.hash_pandas_object (pandas' xxhash-based row hash).
    # Polars has no direct equivalent; this uses DataFrame.hash_rows(), a different
    # hash algorithm producing different digests for the same data. This is a
    # DELIBERATE BREAKING CHANGE to the checkpoint fingerprint format — any
    # checkpoints on disk from before this migration will be treated as stale
    # (fingerprint mismatch) and the run will restart fresh. See migration report.
    digest = hashlib.sha256()
    digest.update("\0".join(selected).encode("utf-8"))
    digest.update(df.select(selected).hash_rows().to_numpy().tobytes())
    return digest.hexdigest()


def _slug(key):
    """Stable, filesystem-safe id for a stratum name (hashlib, not the salted builtin hash)."""
    return hashlib.md5(str(key).encode('utf-8')).hexdigest()[:16]


def _write_csv_atomic(df, path):
    """Write a CSV beside its destination, then atomically publish the completed file."""
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.checkpoint-', suffix='.csv')
    os.close(fd)
    try:
        df.write_csv(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class RunCheckpoint:
    """Disk-backed checkpoint store for one within-vs-pan run."""

    def __init__(self, checkpoint_dir, fingerprint, resume=True):
        self.dir = checkpoint_dir
        self.fingerprint = fingerprint
        os.makedirs(self.dir, exist_ok=True)

        self.manifest_path = os.path.join(self.dir, 'manifest.csv')
        self.meta_path = os.path.join(self.dir, 'meta.json')
        self.log_path = os.path.join(self.dir, 'progress.log')

        # in-memory manifest: key -> row dict
        self._manifest = {}
        # session bookkeeping for the summary line
        self._resumed, self._fit, self._skipped = set(), set(), set()

        stale = False
        if resume and os.path.exists(self.manifest_path) and os.path.exists(self.meta_path):
            with open(self.meta_path) as fh:
                prev = json.load(fh).get('fingerprint')
            if prev == fingerprint:
                man = pl.read_csv(self.manifest_path, schema_overrides={'key': pl.Utf8})
                self._manifest = {r['key']: dict(r) for r in man.to_dicts()}
                self.log(f"resume: loaded manifest with {len(self._manifest)} entries")
            else:
                stale = True
        if (not resume) or stale or not os.path.exists(self.meta_path):
            if stale:
                self.log("WARNING: checkpoint fingerprint mismatch — ignoring stale checkpoints, starting fresh")
            self._manifest = {}
            with open(self.meta_path, 'w') as fh:
                json.dump({'fingerprint': fingerprint,
                           'created': datetime.now().isoformat(timespec='seconds')}, fh, indent=2)
            self._flush_manifest()

    # ---- logging -------------------------------------------------------------
    def log(self, msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        with open(self.log_path, 'a') as fh:
            fh.write(line + '\n')
        print(line)

    # ---- manifest ------------------------------------------------------------
    def _flush_manifest(self):
        """Rewrite the whole manifest CSV from the in-memory dict.

        Only used to (re)initialize the file — on a fresh/stale-fingerprint start (writes just
        the header, self._manifest is empty at that point) or if a caller ever needs a full
        resync. Per-record writes go through ``_append_manifest_row`` instead (Phase-A A9).
        """
        rows = [{c: row.get(c) for c in MANIFEST_COLS} for row in self._manifest.values()]
        df = pl.DataFrame(rows, schema=MANIFEST_COLS)
        df.write_csv(self.manifest_path)

    def _append_manifest_row(self, row):
        """Append a single record to the manifest CSV without rewriting the whole file.

        The previous implementation rewrote the full CSV on every record, which is O(n^2)
        total writes over a run on a shared filesystem. The manifest is an append-only journal:
        if a key is updated, the last row wins when the next RunCheckpoint is constructed.
        The header is written once, by _flush_manifest(), when the manifest file is initialized.
        """
        write_header = not os.path.exists(self.manifest_path)
        row_df = pl.DataFrame([{c: row.get(c) for c in MANIFEST_COLS}], schema=MANIFEST_COLS)
        with open(self.manifest_path, 'a') as fh:
            row_df.write_csv(fh, include_header=write_header)

    def _record(self, key, status, reason=None, meta=None):
        meta = meta or {}
        row = {c: meta.get(c) for c in MANIFEST_COLS}
        row['key'] = key
        row['status'] = status
        row['reason'] = reason
        if row.get('c_pan') is not None and row.get('c_within') is not None:
            row['delta_c'] = row['c_within'] - row['c_pan']
        row['ts'] = datetime.now().isoformat(timespec='seconds')
        self._manifest[key] = row
        self._append_manifest_row(row)

    def status(self, key):
        """Return 'done' | 'skipped' | None for a stratum key."""
        row = self._manifest.get(key)
        return row['status'] if row else None

    def metadata(self, key):
        """Return a copy of the latest manifest row for a key."""
        row = self._manifest.get(key)
        return dict(row) if row else {}

    def stratum_done(self, key):
        """True only when the manifest and both score files are complete."""
        return (
            self.status(key) == 'done'
            and os.path.isfile(self._train_path(key))
            and os.path.isfile(self._held_path(key))
        )

    # ---- file paths ----------------------------------------------------------
    def _train_path(self, key):
        return os.path.join(self.dir, f'pan_train_scores.csv' if key == PAN_KEY
                            else f'stratum_{_slug(key)}_train.csv')

    def _held_path(self, key):
        return os.path.join(self.dir, f'pan_held_scores.csv' if key == PAN_KEY
                            else f'stratum_{_slug(key)}_held.csv')

    # ---- pan stage -----------------------------------------------------------
    def pan_done(self):
        return self.stratum_done(PAN_KEY)

    def load_pan(self):
        self._resumed.add(PAN_KEY)
        self.log("resumed: pan model scores")
        return pl.read_csv(self._train_path(PAN_KEY)), pl.read_csv(self._held_path(PAN_KEY))

    def save_pan(self, train_scores_df, held_scores_df, meta=None):
        _write_csv_atomic(train_scores_df, self._train_path(PAN_KEY))
        _write_csv_atomic(held_scores_df, self._held_path(PAN_KEY))
        self._record(PAN_KEY, 'done', meta=meta)
        self._fit.add(PAN_KEY)
        m = meta or {}
        self.log(f"fit: pan model (l1={m.get('l1')}, alpha={m.get('alpha')}, "
                 f"n_train={m.get('n_train')}, n_held={m.get('n_held')})")

    # ---- within stage --------------------------------------------------------
    def load_stratum(self, key):
        self._resumed.add(key)
        self.log(f"resumed: {key}")
        return pl.read_csv(self._train_path(key)), pl.read_csv(self._held_path(key))

    def save_stratum(self, key, train_scores_df, held_scores_df, meta=None):
        # Publish both data files before the manifest says this stage is complete. A crash can
        # therefore leave an ignored orphan file, never a resumable record pointing at a
        # half-written CSV.
        _write_csv_atomic(train_scores_df, self._train_path(key))
        _write_csv_atomic(held_scores_df, self._held_path(key))
        self._record(key, 'done', meta=meta)
        self._fit.add(key)
        m = meta or {}
        self.log(f"fit: {key} (n_train={m.get('n_train')}, n_held={m.get('n_held')}, "
                 f"l1={m.get('l1')}, alpha={m.get('alpha')}, "
                 f"c_pan={m.get('c_pan')}, c_within={m.get('c_within')}, "
                 f"{m.get('elapsed_s')}s)")

    def update_stratum_meta(self, key, meta):
        """Add metadata after both independently checkpointed halves are available."""
        if not self.stratum_done(key):
            raise ValueError(f"Cannot update incomplete stratum checkpoint: {key}")
        current = self.metadata(key)
        merged = {column: current.get(column) for column in MANIFEST_COLS}
        merged.update(meta)
        self._record(key, 'done', meta=merged)

    def mark_skipped(self, key, reason, meta=None):
        self._record(key, 'skipped', reason=reason, meta=meta)
        self._skipped.add(key)
        self.log(f"skipped: {key} ({reason})")

    # ---- summary -------------------------------------------------------------
    def counts(self):
        """(#resumed, #fit, #skipped) within strata, excluding matched-pan stages."""
        def within_only(keys):
            return {key for key in keys
                    if key != PAN_KEY and not str(key).endswith('__matched_pan__')}

        return (len(within_only(self._resumed)),
                len(within_only(self._fit)),
                len(within_only(self._skipped)))


class LandmarkCheckpoint:
    """Disk-backed checkpoint store for one mortality-trajectory run.

    ``generate_mortality_trajectories`` scores ~21 landmarks, each costing an embedding re-pool
    plus a nested CV fit, and previously had no resume: an interrupted run restarted from month 0
    and discarded every completed landmark. This store writes each landmark's risk scores as soon
    as they exist, so a re-run reloads them and refits only what is missing.

    Unlike ``RunCheckpoint`` a landmark has no train/held split — one score column per landmark is
    the whole result — so the record is a single CSV keyed by month, not a pair.

    Resume is sound for the same reason it is there: the pipeline is deterministic. The at-risk set
    at a landmark is a fixed function of the baseline event times, pooling is deterministic given
    ``decay_param`` and the window, and the nested CV uses a fixed ``random_state=1234``. Refitting
    a landmark would reproduce the scores being reloaded.

    A failed landmark is recorded as 'failed' rather than left absent, so a resumed run does not
    silently retry a fit that will fail the same way. Pass ``retry_failed=True`` to refit those.

    Besides the per-landmark scores this store holds one small extra record: the month-0
    hyperparameters (``save_hyperparams``/``load_hyperparams``). They are the output of a grid
    search that is separate from — and additional to — the month-0 score column, so without them
    a resume still paid for the search before it could refit the model every landmark scores
    with. Two floats are enough because that model is refit from them cheaply.
    """

    def __init__(self, checkpoint_dir, fingerprint, resume=True, retry_failed=False):
        self.dir = checkpoint_dir
        self.fingerprint = fingerprint
        self.retry_failed = retry_failed
        os.makedirs(self.dir, exist_ok=True)

        self.manifest_path = os.path.join(self.dir, 'landmark_manifest.csv')
        self.meta_path = os.path.join(self.dir, 'landmark_meta.json')
        self.log_path = os.path.join(self.dir, 'landmark_progress.log')

        self._manifest = {}
        self._resumed, self._fit, self._failed = set(), set(), set()
        self._resumed_hyperparams = False

        stale = False
        if resume and os.path.exists(self.manifest_path) and os.path.exists(self.meta_path):
            with open(self.meta_path) as fh:
                prev = json.load(fh).get('fingerprint')
            if prev == fingerprint:
                man = pl.read_csv(self.manifest_path, schema_overrides={'month': pl.Int64})
                self._manifest = {int(r['month']): dict(r) for r in man.to_dicts()}
                done = sum(1 for r in self._manifest.values() if r['status'] == 'done')
                self.log(f"resume: loaded manifest with {len(self._manifest)} entries ({done} done)")
            else:
                stale = True
        if (not resume) or stale or not os.path.exists(self.meta_path):
            if stale:
                self.log("WARNING: checkpoint fingerprint mismatch — ignoring stale checkpoints, "
                         "starting fresh")
            self._manifest = {}
            # The hyperparameters are derived from the same cohort the fingerprint covers, so
            # they are only valid alongside the manifest being cleared here.
            self._reset_hyperparams()
            with open(self.meta_path, 'w') as fh:
                json.dump({'fingerprint': fingerprint,
                           'created': datetime.now().isoformat(timespec='seconds')}, fh, indent=2)
            pl.DataFrame([], schema=LANDMARK_MANIFEST_COLS).write_csv(self.manifest_path)

    # ---- logging -------------------------------------------------------------
    def log(self, msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        with open(self.log_path, 'a') as fh:
            fh.write(line + '\n')
        print(line, flush=True)

    # ---- manifest ------------------------------------------------------------
    def _append_manifest_row(self, row):
        """Append one record without rewriting the file (same rationale as RunCheckpoint's)."""
        write_header = not os.path.exists(self.manifest_path)
        row_df = pl.DataFrame([{c: row.get(c) for c in LANDMARK_MANIFEST_COLS}],
                              schema=LANDMARK_MANIFEST_COLS)
        with open(self.manifest_path, 'a') as fh:
            row_df.write_csv(fh, include_header=write_header)

    def _record(self, month, status, reason=None, meta=None):
        meta = meta or {}
        row = {c: meta.get(c) for c in LANDMARK_MANIFEST_COLS}
        row['month'] = int(month)
        row['status'] = status
        row['reason'] = reason
        row['ts'] = datetime.now().isoformat(timespec='seconds')
        self._manifest[int(month)] = row
        self._append_manifest_row(row)

    def _scores_path(self, month):
        return os.path.join(self.dir, f'landmark_{int(month):03d}_scores.csv')

    # ---- month-0 hyperparameters ---------------------------------------------
    # The month-0 grid search is a separate expense from the month-0 score column and was the
    # one un-checkpointed stage left: a resume reloaded the nested-CV scores but still spent the
    # full grid search re-deriving the two floats below, because only CSVs of scores were ever
    # persisted (never the fitted object).  The model carried to every landmark is refit from
    # these two numbers in ~90s, so storing them is enough to skip the search entirely.
    #
    # Staleness is handled by the same fingerprint gate as the manifest: this file lives in the
    # checkpoint dir, and a fingerprint mismatch clears the dir's manifest and rewrites meta,
    # so a stale hyperparameter file is discarded with everything else (see _reset_hyperparams).
    def _hyperparams_path(self):
        return os.path.join(self.dir, 'month0_hyperparams.json')

    def load_hyperparams(self):
        """Reload the checkpointed month-0 (l1_ratio, alpha, mean_auc), or None.

        Returns None whenever the file is absent, unreadable or missing either hyperparameter,
        so a corrupt file costs a grid search rather than a wrong model.
        """
        path = self._hyperparams_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as fh:
                rec = json.load(fh)
            l1, alpha = float(rec['l1_ratio']), float(rec['alpha'])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.log("month-0 hyperparameter checkpoint unreadable — rerunning the grid search")
            return None
        auc = rec.get('mean_auc')
        self._resumed_hyperparams = True
        self.log(f"resumed: month-0 hyperparameters (l1_ratio={l1}, alpha={alpha:.3e})")
        return l1, alpha, (float(auc) if auc is not None else None)

    def save_hyperparams(self, l1_ratio, alpha, mean_auc=None, elapsed_s=None):
        """Persist the selected month-0 hyperparameters for later resumes."""
        payload = {'l1_ratio': float(l1_ratio), 'alpha': float(alpha),
                   'mean_auc': None if mean_auc is None else float(mean_auc),
                   'elapsed_s': None if elapsed_s is None else round(float(elapsed_s), 1),
                   'ts': datetime.now().isoformat(timespec='seconds')}
        path = self._hyperparams_path()
        fd, tmp_path = tempfile.mkstemp(dir=self.dir, prefix='.hyperparams-', suffix='.json')
        try:
            with os.fdopen(fd, 'w') as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        self.log(f"saved: month-0 hyperparameters (l1_ratio={payload['l1_ratio']}, "
                 f"alpha={payload['alpha']:.3e})")

    def _reset_hyperparams(self):
        """Drop a stale/ignored hyperparameter checkpoint alongside the manifest reset."""
        path = self._hyperparams_path()
        if os.path.exists(path):
            os.unlink(path)

    # ---- query ---------------------------------------------------------------
    def status(self, month):
        """Return 'done' | 'failed' | None for a landmark month."""
        row = self._manifest.get(int(month))
        return row['status'] if row else None

    def landmark_done(self, month):
        """True only when the manifest says done AND the score file is on disk."""
        return self.status(month) == 'done' and os.path.isfile(self._scores_path(month))

    def failure_reason(self, month):
        """Recorded reason for a failed landmark, or None."""
        row = self._manifest.get(int(month))
        return row.get('reason') if row and row['status'] == 'failed' else None

    def should_skip_failed(self, month):
        """True when a previously failed landmark should not be retried this run."""
        return self.status(month) == 'failed' and not self.retry_failed

    # ---- records -------------------------------------------------------------
    def load_landmark(self, month):
        """Reload a completed landmark's scores as a {DFCI_MRN: risk_score} dict."""
        self._resumed.add(int(month))
        df = pl.read_csv(self._scores_path(month))
        self.log(f"resumed: month {month} ({df.height} scored patients)")
        return dict(zip(df['DFCI_MRN'].to_list(), df['risk_score'].to_list()))

    def save_landmark(self, month, scores_df, meta=None):
        """Persist one landmark's scores. Written before the manifest row, so a crash
        mid-write leaves an unreferenced file rather than a manifest entry with no data."""
        _write_csv_atomic(scores_df.select(['DFCI_MRN', 'risk_score']),
                          self._scores_path(month))
        self._record(month, 'done', meta=meta)
        self._fit.add(int(month))
        m = meta or {}
        self.log(f"fit: month {month} (n_at_risk={m.get('n_at_risk')}, "
                 f"n_scored={m.get('n_scored')}, {m.get('elapsed_s')}s)")

    def mark_failed(self, month, reason, meta=None):
        self._record(month, 'failed', reason=reason, meta=meta)
        self._failed.add(int(month))
        self.log(f"failed: month {month} ({reason})")

    # ---- summary -------------------------------------------------------------
    def counts(self):
        """(#resumed, #fit, #failed) landmarks this session."""
        return (len(self._resumed), len(self._fit), len(self._failed))

    def recorded_failures(self):
        """[(month, reason)] for every landmark marked failed, this session or a previous one."""
        return sorted(
            (int(r['month']), r.get('reason'))
            for r in self._manifest.values() if r['status'] == 'failed'
        )
