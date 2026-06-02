"""Resumable checkpointing for long-running within-vs-pan model scripts.

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
import time
from datetime import datetime

import pandas as pd

PAN_KEY = '__pan__'
MANIFEST_COLS = ['key', 'status', 'reason', 'n_train', 'n_held',
                 'l1', 'alpha', 'c_pan', 'c_within', 'delta_c', 'elapsed_s', 'ts']


def _slug(key):
    """Stable, filesystem-safe id for a stratum name (hashlib, not the salted builtin hash)."""
    return hashlib.md5(str(key).encode('utf-8')).hexdigest()[:16]


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
                man = pd.read_csv(self.manifest_path, dtype={'key': str})
                self._manifest = {r['key']: dict(r) for r in man.to_dict('records')}
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
        df = pd.DataFrame(list(self._manifest.values()), columns=MANIFEST_COLS)
        df.to_csv(self.manifest_path, index=False)

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
        self._flush_manifest()

    def status(self, key):
        """Return 'done' | 'skipped' | None for a stratum key."""
        row = self._manifest.get(key)
        return row['status'] if row else None

    # ---- file paths ----------------------------------------------------------
    def _train_path(self, key):
        return os.path.join(self.dir, f'pan_train_scores.csv' if key == PAN_KEY
                            else f'stratum_{_slug(key)}_train.csv')

    def _held_path(self, key):
        return os.path.join(self.dir, f'pan_held_scores.csv' if key == PAN_KEY
                            else f'stratum_{_slug(key)}_held.csv')

    # ---- pan stage -----------------------------------------------------------
    def pan_done(self):
        return self.status(PAN_KEY) == 'done'

    def load_pan(self):
        self._resumed.add(PAN_KEY)
        self.log("resumed: pan model scores")
        return pd.read_csv(self._train_path(PAN_KEY)), pd.read_csv(self._held_path(PAN_KEY))

    def save_pan(self, train_scores_df, held_scores_df, meta=None):
        train_scores_df.to_csv(self._train_path(PAN_KEY), index=False)
        held_scores_df.to_csv(self._held_path(PAN_KEY), index=False)
        self._record(PAN_KEY, 'done', meta=meta)
        self._fit.add(PAN_KEY)
        m = meta or {}
        self.log(f"fit: pan model (l1={m.get('l1')}, alpha={m.get('alpha')}, "
                 f"n_train={m.get('n_train')}, n_held={m.get('n_held')})")

    # ---- within stage --------------------------------------------------------
    def load_stratum(self, key):
        self._resumed.add(key)
        self.log(f"resumed: {key}")
        return pd.read_csv(self._train_path(key)), pd.read_csv(self._held_path(key))

    def save_stratum(self, key, train_scores_df, held_scores_df, meta=None):
        train_scores_df.to_csv(self._train_path(key), index=False)
        held_scores_df.to_csv(self._held_path(key), index=False)
        self._record(key, 'done', meta=meta)
        self._fit.add(key)
        m = meta or {}
        self.log(f"fit: {key} (n_train={m.get('n_train')}, n_held={m.get('n_held')}, "
                 f"l1={m.get('l1')}, alpha={m.get('alpha')}, "
                 f"c_pan={m.get('c_pan')}, c_within={m.get('c_within')}, "
                 f"{m.get('elapsed_s')}s)")

    def mark_skipped(self, key, reason, meta=None):
        self._record(key, 'skipped', reason=reason, meta=meta)
        self._skipped.add(key)
        self.log(f"skipped: {key} ({reason})")

    # ---- summary -------------------------------------------------------------
    def counts(self):
        """(#resumed, #fit, #skipped) strata this session, excluding the pan stage."""
        return (len(self._resumed - {PAN_KEY}),
                len(self._fit - {PAN_KEY}),
                len(self._skipped))
